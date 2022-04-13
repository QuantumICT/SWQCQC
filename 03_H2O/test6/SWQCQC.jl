using MPI
MPI.Init()
const root = 0
push!(LOAD_PATH, "../../QuantumSpin/src")
using QuantumSpins
using NLopt

# By adding this macro to print/println, flushing will be done in any IOStream.
# The @auto_flush is already applied to print/println.
# DO NOT add this macro to print/println again.
macro auto_flush(p)
	if !(p isa Expr) || !(p.args[1] in [:print,:println,:(Base.print),:(Base.println)])
		return :($(esc(p)))
	end
	quote
		local io = $(esc(Expr(:ref, Expr(:tuple, p.args[2]), 1)))
		$(esc(p))
		if !(io isa IO)
			flush(stdout)
		else
			flush(io)
		end
	end
end

# Now default print/println is auto-flushing, there is NO need to manually add @auto_flush.
print(args...) = @auto_flush Base.print(args...)
println(args...) = @auto_flush Base.println(args...)

# Another version of @time. Output is redirected to io.
macro time_redirect(io, ex)
	quote
		local t0 = time_ns()
		local val = $(esc(ex))
		local t1 = time_ns()
		println($(esc(io)), "elapsed time: ", (t1-t0)/1e9, " seconds")
		val
	end
end

function p_range(n::Int, n_procs::Int, pid::Int)::UnitRange{Int}
	aprocs = n_procs - n % n_procs + 1
	q = n ÷ n_procs
	if (pid < aprocs)
			pstart = (pid - 1) * q + 1
			pend = pstart + q - 1
	else
			pstart = (aprocs-1) * q + (pid - aprocs)*(q+1) + 1
			pend = pstart + q
	end
	return pstart:pend
end

const QubitOperator = Dict{Tuple{Vararg{Tuple{Int64, String}}}, ComplexF64}

function MPI.Scatterv(QO :: Union{Nothing, QubitOperator}, root :: Integer, comm::MPI.Comm)
	if root == MPI.Comm_rank(comm)
			@assert QO != nothing
			len = QO.count
			n_procs = MPI.Comm_size(comm)
			QO_pairs = collect(QO)
			reqs = Vector{MPI.Request}(undef, n_procs - 1)
			for i in 1:n_procs-1
					qo = QubitOperator(@view QO_pairs[p_range(len, n_procs, i+1)])
					reqs[i] = MPI.isend(qo, i, 0, comm)
			end
			MPI.Waitall!(reqs)
			return QubitOperator(@view QO_pairs[p_range(len, n_procs, 1)])
	else
			qo, status = MPI.recv(root, 0, comm)
			return qo
	end
end

const hy_matrix = [+1.0+0.0im +0.0-1.0im; 0.0+1.0im -1.0+0.0im] * 0.5 * 2^0.5
function gates_apply_dict(qb :: AbstractString, idx :: Int) :: Union{Nothing, HGate, QuantumGate}
	if qb == "X"
		return HGate(idx + 1)
	elseif qb == "Y"
		return Gate(idx + 1, hy_matrix)
	else
		return nothing
	end
end
function hadamard_test_dict(term :: Tuple{Int64, String}, n_qubits :: Int) :: Vector{Union{CNOTGate, CZGate, RzGate}}
	if term[2] == "X"
		return [CNOTGate(n_qubits, term[1] + 1)]
	elseif term[2] == "Y"
		return [RzGate(term[1] + 1, -pi/2), CNOTGate(n_qubits, term[1] + 1), RzGate(term[1] + 1, pi/2)]
	else
		return [CZGate(n_qubits, term[1] + 1)]
	end
end
 
function decompose_trottered_qubit_op!(QC :: QuantumCircuit, qubit_op :: QubitOperator, amp::Int)
	

	for (terms,coeff) in qubit_op

		for (idx, qubit_gate) in terms
			gate = gates_apply_dict(qubit_gate, idx)
			if gate != nothing
				push!(QC, gate)
			end
		 				
		end
		for i in length(terms)-1:-1:1
			push!(QC, CNOTGate(terms[i+1][1] + 1, terms[i][1] + 1))
		end
		push!(QC, AmpRzGate(terms[1][1] + 1, -2*imag(coeff), amp))
		for i in 1:length(terms)-1
			push!(QC, CNOTGate(terms[i+1][1] + 1, terms[i][1] + 1))
		end
		for (idx, qubit_gate) in terms
			gate = gates_apply_dict(qubit_gate, idx)
			if gate != nothing
				push!(QC, gate)
			end
		 				
		end
	end
	
end

function add_HF_circuit!(QC :: QuantumCircuit, occ_indices_spin :: Vector{Int})
	for i in occ_indices_spin
		push!(QC, XGate(i + 1))
	end
end

function add_U_circuit!(QC :: QuantumCircuit, n :: Int, ucc_operator_pool_qubit_op::Vector{QubitOperator})
	for idx in 1:n
		decompose_trottered_qubit_op!(QC, ucc_operator_pool_qubit_op[idx], idx)
	end
end
 
function construct_S2(h_terms :: Tuple{Vararg{Tuple{Int64, String}}}, n_qubits :: Int) :: QuantumCircuit
	QC = QuantumCircuit()
	push!(QC, HGate(n_qubits))
	for term in h_terms	
		for gate in hadamard_test_dict(term, n_qubits)
			push!(QC, gate)
		end
	end	
	push!(QC, HGate(n_qubits))
	return QC
end

function generate_trottered_ansatz_circuit!(QC :: QuantumCircuit, amplitudes :: Vector{Float64})
	for i in 1:length(QC)
		g = getindex(QC, i)
		if (typeof(g)==AmpRzGate{Float64})
			setindex!(QC, RzGate(g.positions[1], g.parameter * amplitudes[g.amp]), i)
		end
	end
end

function get_expec(HFU :: QuantumCircuit, S2 :: Vector{QuantumCircuit}, amplitudes :: Vector{Float64}, n_qubits :: Int) :: Vector{Float64}
	trottered_HFU = copy(HFU)
	generate_trottered_ansatz_circuit!(trottered_HFU, amplitudes)
	wave_function::AbstractMPS = statevector_mps(n_qubits)
	trunc = MPSTruncation(ϵ=1.0e-5, D=128)
	apply!(trottered_HFU, wave_function, trunc=trunc)
	
	expecs = fill(1.0, length(S2))
	for i in 1:length(S2)
		state::AbstractMPS = copy(wave_function)
		apply!(S2[i], state, trunc=trunc)
		outcome, prob_at_measure_qubit = measure!(state, n_qubits, keep=true)
		if outcome == 0
			expecs[i] = 2 * prob_at_measure_qubit - 1
		else outcome == 1
			expecs[i] = 2 * (1 - prob_at_measure_qubit) - 1
		end
	end
	return expecs
end

function vqe(parameters :: Dict, comm :: MPI.Comm) #:: Union{Nothing, AbstractMPS}

	rank = MPI.Comm_rank(comm)
	nprocs = MPI.Comm_size(comm)
	

	n_qubits::Int = parameters["n_qubits"]	# NOT including auxilary qubit!!!
	n_amplitudes::Int = parameters["n_params"]
	spin_orbital_occupied_indices::Vector{Int} = parameters["spin_orbital_occupied_indices"]
	ucc_operator_pool_qubit_op::Vector{QubitOperator} = parameters["ucc_operator_pool_qubit_op"]	
	hamiltonian_qubit_op::QubitOperator = parameters["hamiltonian_qubit_op"]
	n_circuits = length(hamiltonian_qubit_op)

	HFU = QuantumCircuit()
	add_HF_circuit!(HFU, spin_orbital_occupied_indices)
	add_U_circuit!(HFU, n_amplitudes, ucc_operator_pool_qubit_op)

	# static task(circuit) split
	n_qubits = n_qubits + 1		#NOW including the auxilary qubit !!!
	S2 = Vector{QuantumCircuit}()
	h = Vector{Float64}()
	for (k, v) in hamiltonian_qubit_op
		push!(h, real(v))
		push!(S2, construct_S2(k, n_qubits))
	end
	
	#so far, we have same HF+U on each process, and different S2 for each circuit 

	if rank == root
		
		function get_total_energy(amplitudes :: Vector{Float64}) :: Float64
			MPI.bcast(amplitudes, root, comm)
			energy = sum(h.*get_expec(HFU, S2, amplitudes, n_qubits))
			res::Float64 = MPI.Reduce(energy, +, root, comm)	
			println(out_file, "--------------end get_total_energy----------  ",res)
			return res
		end
		function finalize()
			MPI.bcast(nothing, root, comm)
		end
	else
		function work()
			while (true)
				amplitudes::Union{Nothing, Vector{Float64}} = MPI.bcast(nothing, root, comm)
				if amplitudes == nothing
					break
				end
				energy = sum(h.*get_expec(HFU, S2, amplitudes, n_qubits))
				MPI.Reduce(energy, +, root, comm)
			end			
		end
	end

	# start VQE optimization
	if rank == root
		for i in 1:1
			@time_redirect out_file get_total_energy(fill(0.0, n_amplitudes))
		end
		opt = Opt(:LN_BOBYQA, n_amplitudes)
		opt.min_objective = (amplitudes::Vector{Float64}, grad::Vector{Float64}) -> @time_redirect out_file get_total_energy(amplitudes)
		opt.maxeval = 20
		opt.xtol_rel = 1e-7
     	t1=time()
     	(minf, minx, ret) = optimize(opt, fill(0.0, n_amplitudes))
		t2=time()
		println(out_file, "elapsed time: ",t2-t1," seconds, got $minf at $minx (returned $ret)")
		
		finalize()
	else
		work()
	end	

	#	if rank == root
	#		MPI.bcast(minx, root, comm)
	#	else
	#		minx::Vector{Float64} = MPI.bcast(nothing, root, comm)
	#	end

	#	generate_trottered_ansatz_circuit!(HFU, minx)
	#	vqe_state::AbstractMPS = statevector_mps(n_qubits)
	#	apply!(HFU, vqe_state, trunc=MPSTruncation(ϵ=1.0e-5, D=128))
	#	return vqe_state

end

function read_binary_qubit_op(f::IO, n_qubits::Int) :: QubitOperator

    pauli_symbol_dict = Dict(
        0 => "I",
        1 => "X",
        2 => "Y",
        3 => "Z"
    )
	qubit_op_dict = QubitOperator()
	len = read(f, Int64)
	if len > 0			# paulis are stored in dense array
		for i in 1:len
			coeff_tmp = read(f, ComplexF64)
			pauli_str_tmp = Int8[read(f, Int8) for i in 1:n_qubits]
			pauli_str_tuple = Tuple([(i - 1, pauli_symbol_dict[pauli_str_tmp[i]])
                                 for i in 1:n_qubits
                                 if pauli_str_tmp[i] != 0])
			qubit_op_dict[pauli_str_tuple] = coeff_tmp
		end
	else				# paulis are stored in compressed array
		len = -len
		for i in 1:len
			coeff_tmp = read(f, ComplexF64)
			len_pauli_array = read(f, Int32)
			pauli_str_tuple = Vector{Tuple{Int64, String}}()
			for i in 1:len_pauli_array
				push!(pauli_str_tuple, (read(f, Int32), pauli_symbol_dict[read(f, Int8)]))
			end
			qubit_op_dict[Tuple(pauli_str_tuple)] = coeff_tmp
		end
	end
	return qubit_op_dict
end

function read_binary_dict(f::IO)::Dict{String,Any}

    identifier = read(f, Float64)
    err_msg = identifier != Float64(99.0212) && error("The file is not saved as vqe parameters.")

	d = Dict{String, Any}()
	d["n_qubits"] = Int64(read(f, Int32))
	len_spin_indices = read(f, Int32)
	d["spin_orbital_occupied_indices"] = Int64[read(f,Int32) for i in 1:len_spin_indices]
	d["n_params"] = Int64(read(f, Int32))
	d["ucc_operator_pool_qubit_op"] = Vector{QubitOperator}(undef, d["n_params"])
	for i in 1:d["n_params"]
		d["ucc_operator_pool_qubit_op"][i] = read_binary_qubit_op(f, d["n_qubits"])
	end
	return d
end
xzqsize(x) = Base.format_bytes(Base.summarysize(x))

function run_task(comm :: MPI.Comm)
	rank = MPI.Comm_rank(comm)

	p = Dict{String, Any}()
	if rank == root
		f = open(ARGS[1], "r")

		p = read_binary_dict(f)

		@time_redirect out_file MPI.bcast(p, root, comm)
		println(out_file, "Broadcasting parameters finished!")

		H = read_binary_qubit_op(f, p["n_qubits"])
		close(f)

		p["hamiltonian_qubit_op"] = MPI.Scatterv(H, root, comm)

		println(out_file, "n_amplitudes : $(p["n_params"])")
		println(out_file, "n_circuits : $(H.count)")
	else
		p = MPI.bcast(nothing, root, comm)
		p["hamiltonian_qubit_op"] = MPI.Scatterv(nothing, root, comm)
	end
	
	vqe(p,comm)

end

function main()

	old_comm = MPI.COMM_WORLD
	old_rank = MPI.Comm_rank(old_comm)
	old_nprocs = MPI.Comm_size(old_comm)
	if length(ARGS) > 1
		group_nprocs = parse(Int64, ARGS[2])
	else
		group_nprocs = old_nprocs
	end

	color = Int(trunc(old_rank / group_nprocs))
	new_comm = MPI.Comm_split(old_comm, color, old_rank)
	new_rank = MPI.Comm_rank(new_comm)
	new_nprocs = MPI.Comm_size(new_comm)

	if new_rank == root
		global out_file = open("out"*string(color), "w")
		println("task $color start")
	end

	t1 = time()
	run_task(new_comm)
	t2 = time()

	if new_rank == root
		println("task $color finish, time: $(t2-t1) seconds")
	end
	
end
main()
