
const ProjMPO_general = Union{ProjMPO,ITensors.DiskProjMPO}

macro timeit(args...)
  # Redefine the behavior of the @timeit macro here
  return quote end
end

function dmrgX(x1, psi0::MPS, targetPsi::MPS; kwargs...)
  return dmrgX(x1, psi0, targetPsi, _dmrg_sweeps(; kwargs...); kwargs...)
end

function dmrgX(x1, targetPsi::MPS; kwargs...)
  psi0 = deepcopy(targetPsi)
  return dmrgX(x1, psi0, targetPsi, _dmrg_sweeps(; kwargs...); kwargs...)
end

function dmrgX(H::MPO, psi0::MPS, targetPsi::MPS, sweeps::Sweeps; kwargs...)
  check_hascommoninds(siteinds, H, psi0)
  check_hascommoninds(siteinds, H, psi0')
  # Permute the indices to have a better memory layout
  # and minimize permutations
  H = permute(H, (linkind, siteinds, linkind))
  PH = ProjMPO(H)
  return dmrgX(PH, psi0, targetPsi, sweeps; kwargs...)
end

function extract_bitstring(psi::MPS)
  bitstring = zeros(length(psi))
  for i in 1:length(psi)
    # println(vec(psi[i].tensor))

    bitstring[i] = findall(x -> abs(x) == 1.0, vec(psi[i].tensor))[1][1]
  end
  return bitstring
end

function get_amplitude(psi::MPS, bitstring)
  bitstring = Int.(bitstring)
  N = length(psi)
  # overlap = 1.0
  # for i in 1:N
  #   bool_idcs = inds(state[i]) .== siteind(state, i)

  #   indexing_tuple = ntuple(length(bool_idcs)) do i
  #     bool_idcs[i] ? bitstring[i] + 1 : Colon()
  #   end
  #   if i == 1
  #     overlap *= state[i].tensor[indexing_tuple...]'
  #   else
  #     overlap *= state[i].tensor[indexing_tuple...]
  #   end
  #   println(overlap)
  # end

  overlap = ITensor(1.0)
  for j in 1:N
    site = siteind(psi, j)

    overlap *= (psi[j] * state(site, bitstring[j]))
  end
  overlap = scalar(overlap)
  return overlap
end

function get_amplitude2(psi::MPS, eigenvector, bitstring, b, left, right)
  N = length(psi)

  overlap = ITensor(1.0)
  # for j in 1:N
  #   site = siteind(psi, j)
  #   # println(inds(psi[j]))
  #   if j == b
  # println("\n eigenvector: ", eigenvector, "\n")
  # println(inds(eigenvector))
  overlap *= (
    eigenvector *
    state(siteind(psi, b), bitstring[b]) *
    state(siteind(psi, b + 1), bitstring[b + 1])
  )
  overlap *= left * right
  # elseif j == b + 1
  #   continue
  # else
  #   overlap *= (psi[j] * state(site, bitstring[j]))
  # end
  # println("\n overlap: ", overlap, "\n")
  # end
  overlap = scalar(overlap)
  return overlap
end

function get_amplitude3(psi::MPS, eigenvector, bitstring, b, left, right)
  N = length(psi)

  overlap = ITensor(1.0)

  overlap *= extract_tensor(eigenvector, bitstring[[b, b + 1]], siteinds(psi)[[b, b + 1]])
  overlap *= left * right

  overlap = scalar(overlap)
  return overlap
end

function extract_tensor(T::ITensor, bit::Int, target_ind::Index)
  return extract_tensor(T, [bit], [target_ind])
end
function extract_tensor(T::ITensor, bits::Vector{Int}, target_inds::Vector)
  if length(bits) != length(target_inds)
    throw(ArgumentError("Length of bits and target_inds must be equal"))
  end

  t_inds = inds(T)
  extract_indices = [
    in(ind, target_inds) ? bits[findfirst(isequal(ind), target_inds)] : Colon() for
    ind in t_inds
  ]

  other_inds = setdiff(t_inds, target_inds)
  return ITensor(T.tensor[extract_indices...], other_inds)
end

function dmrg_x_solver_new(
  PH::ProjMPO,
  psi0::MPS,
  target::MPS,
  b::Int,
  sweeps,
  sw,
  ortho,
  which_decomp,
  svd_alg;
  limits::Vector{}=[],
  is_gs::Bool=false,
)
  # println(b)
  H = contract(PH, ITensor(true))
  # println(inds(H))
  time_diag = @elapsed begin
    D, U = eigen(H; ishermitian=true)
  end
  # println("-----\n Diagonalization time: ", time_diag)
  #println(cutoff)
  U_inds = inds(U)[1:(end - 1)]
  eig_inds = inds(U)[end]
  psi = deepcopy(psi0)

  overlaps = zeros(eig_inds.space)
  # println("EIG INDS: ", eig_inds)
  sum_overlap = 0.0
  max_ind = eig_inds.space
  target_bitstring = extract_bitstring(target)
  target_bitstring = Int.(target_bitstring)

  is_growing = abs(D[1, 1]) < abs(D[end, end])
  iteration_range = is_growing ? (1:(eig_inds.space)) : (eig_inds.space):-1:1

  #Definition of left and right contractions that are not touched by the setcutoff
  left = ITensor(1.0)
  for j in 1:(b - 1)
    site = siteind(psi, j)
    left *= extract_tensor(psi[j], target_bitstring[j], site)
    #left *= (psi[j] * state(site, target_bitstring[j]))
  end
  right = ITensor(1.0)
  for j in (b + 2):length(psi)
    site = siteind(psi, j)
    right *= extract_tensor(psi[j], target_bitstring[j], site)
    # println(right)
    #right *= (psi[j] * state(site, target_bitstring[j]))
  end

  overlaps_time = @elapsed begin
    for i in iteration_range
      #X is the eigenvectors
      ind = eig_inds[i]
      eigenvector = U * onehot(ComplexF64, ind)

      # println("------\n TARGET BITSTRING: ", target_bitstring, "\n")
      # time_old = @elapsed overlap_old = get_amplitude(psi, target_bitstring)
      # time_new = @elapsed overlap = get_amplitude2(psi, eigenvector, target_bitstring, b)
      overlap = get_amplitude3(psi, eigenvector, target_bitstring, b, left, right)
      # println("OVERLAP: ", abs(overlap), " OLD: ", abs(overlap_old))
      # println(i)
      # println("TIME OLD: ", time_old, " TIME NEW: ", time_new)
      overlaps[ind.second] = abs(overlap)^2

      sum_overlap += overlaps[ind.second]

      if overlaps[ind.second] > overlaps[max_ind]
        max_ind = ind.second
      end

      if overlaps[max_ind] > (1 - sum_overlap)
        global max_overlap = overlaps[max_ind]
        # N_iterations = is_growing ? i : eig_inds.space - i
        # println("N_iterations: ", N_iterations)
        global spec = replacebond!(
          psi,
          b,
          eigenvector;
          maxdim=maxdim(sweeps, sw),
          mindim=mindim(sweeps, sw),
          cutoff=cutoff(sweeps, sw),
          ortho=ortho,
          normalize=true,
          which_decomp=which_decomp,
          svd_alg=svd_alg,
        )
        break
      end
    end
  end

  u = uniqueind(U, H)

  U_max = 0
  eig = 0

  U_max = U * dag(onehot(u => max_ind))
  eig = D[max_ind, max_ind]

  return U_max, eig, spec, psi
end

function dmrg_x_solver(
  PH::ProjMPO,
  psi0::MPS,
  target::MPS,
  b::Int,
  sweeps,
  sw,
  ortho,
  which_decomp,
  svd_alg;
  limits::Vector{}=[],
  is_gs::Bool=false,
  exact_diag=true,
  phi,
  kwargs...,
)
  gpu = occursin("CUDA", string(typeof(psi0[1].tensor)))
  if exact_diag
    H = contract(PH, ITensor(true))
    H⁺ = swapprime(dag(H), 0 => 1)
    # println(inds(H))
    time_diag = @elapsed begin
      D, U = eigen(0.5 * (H + H⁺); ishermitian=true)
    end
    # println("-----\n Diagonalization time: ", time_diag)
    #println(cutoff)
    U_inds = inds(U)[1:(end - 1)]
    eig_inds = inds(U)[end]
    psi = deepcopy(psi0)

    overlaps = zeros(eig_inds.space)
    # println("EIG INDS: ", eig_inds)
    sum_overlap = 0.0
    max_ind = eig_inds.space

    overlaps_time = @elapsed begin
      is_growing = abs(D[1, 1]) < abs(D[end, end])
      iteration_range = is_growing ? (1:(eig_inds.space)) : (eig_inds.space):-1:1
      N_iterations = 0
      for i in iteration_range
        N_iterations += 1
        #X is the eigenvectors
        ind = eig_inds[i]
        if gpu
          eigenvector = U * NDTensors.cu(onehot(ComplexF64, ind))
        else
          eigenvector = U * onehot(ComplexF64, ind)
        end
        global spec = replacebond!(
          psi,
          b,
          eigenvector;
          maxdim=maxdim(sweeps, sw),
          mindim=mindim(sweeps, sw),
          cutoff=cutoff(sweeps, sw),
          ortho=ortho,
          normalize=true,
          which_decomp=which_decomp,
          svd_alg=svd_alg,
        )

        overlaps[ind.second] = abs(inner(psi, target))^2
        sum_overlap += overlaps[ind.second]
        if overlaps[ind.second] > overlaps[max_ind]
          max_ind = ind.second
        end

        if overlaps[max_ind] > (1 - sum_overlap)
          global max_overlap = overlaps[max_ind]
          # println("max_ind: ", max_ind)
          # println("max overlap: ", max_overlap)
          # println("N_iterations: ", N_iterations)
          break
        end
      end
    end

    u = uniqueind(U, H)

    U_max = 0
    eig = 0

    if gpu
      U_max = U * NDTensors.cu(dag(onehot(u => max_ind)))
    else
      U_max = U * dag(onehot(u => max_ind))
    end
    eig = D[max_ind, max_ind]

    return U_max, eig, spec, psi

  else
    H = contract(PH, ITensor(true))
    H⁺ = swapprime(dag(H), 0 => 1)
    H_alg = 0.5 * (H + H⁺)
    println("Lanczos alg")
    eigsolve_tol = 1e-14
    eigsolve_verbosity = 1

    increase_eig = true
    n_eigs = 0
    while increase_eig
      n_eigs += 1
      println("n_eigs_Lanczos: $n_eigs")
      eigsolve_krylovdim = 2 * n_eigs + 1
      eigsolve_which_eigenvalue = :SR
      eigsolve_maxiter = 10000

      vals, vecs, info = eigsolve(
        H_alg,
        phi,
        n_eigs,
        eigsolve_which_eigenvalue;
        ishermitian=true,
        tol=eigsolve_tol,
        krylovdim=eigsolve_krylovdim,
        maxiter=eigsolve_maxiter,
        verbosity=2,
      )

      # println("Info : $info")
      psi = deepcopy(psi0)

      overlaps = zeros(n_eigs)

      overlaps_time = @elapsed begin
        max_ind = 1
        sum_overlap = 0.0
        for ind in 1:n_eigs
          global spec = replacebond!(
            psi,
            b,
            vecs[ind];
            maxdim=maxdim(sweeps, sw),
            mindim=mindim(sweeps, sw),
            cutoff=cutoff(sweeps, sw),
            ortho=ortho,
            normalize=true,
            which_decomp=which_decomp,
            svd_alg=svd_alg,
          )

          overlaps[ind] = abs(inner(psi, target))^2
          sum_overlap += overlaps[ind]
          if overlaps[ind] > overlaps[max_ind]
            max_ind = ind
          end
          if overlaps[max_ind] > (1 - sum_overlap)
            global max_overlap = overlaps[max_ind]
            println("N_iterations: ", max_ind)
            println("Max overlap: ", max_overlap)
            increase_eig = false
            return vecs[max_ind], vals[max_ind], spec, psi

            # else
            # GC.gc()
          end
        end
      end
    end
    return ErrorException
  end
end

function dmrg_x_solver_ss(
  PH::ProjMPO,
  psi0::MPS,
  target::MPS,
  b::Int,
  sweeps,
  sw,
  ortho,
  which_decomp,
  svd_alg;
  limits::Vector{}=[],
  is_gs::Bool=false,
  exact_diag=true,
  phi,
  kwargs...,
)
  drho = nothing
  pos = copy(b)
  if noise(sweeps, sw) > 0.0
    if pos == length(PH.H)
      pos = b - 1
    end
    chi = psi0[pos] * psi0[pos + 1]
    PH.nsite = 2
    position!(PH, psi0, pos)

    drho = noise(sweeps, sw) * noiseterm(PH, chi, ortho)

    global spec = replacebond!(
      psi0,
      pos,
      chi;
      maxdim=maxdim(sweeps, sw),
      mindim=mindim(sweeps, sw),
      cutoff=cutoff(sweeps, sw),
      eigen_perturbation=drho,
      ortho=ortho,
      normalize=true,
      which_decomp=which_decomp,
      svd_alg=svd_alg,
    )
    PH.nsite = 1
    println("State BD after pertubating it is: ", maxlinkdim(psi0))
    position!(PH, psi0, b)
  end
  H = contract(PH, ITensor(true))
  time_diag = @elapsed begin
    D, U = eigen(H; ishermitian=false)
  end

  U_inds = inds(U)[1:(end - 1)]
  eig_inds = inds(U)[end]
  psi = deepcopy(psi0)
  overlaps = zeros(eig_inds.space)
  # println("EIG INDS: ", eig_inds)
  sum_overlap = 0.0
  max_ind = eig_inds.space

  overlaps_time = @elapsed begin
    is_growing = abs(D[1, 1]) < abs(D[end, end])
    iteration_range = is_growing ? (1:(eig_inds.space)) : (eig_inds.space):-1:1
    N_iterations = 0
    for i in iteration_range
      N_iterations += 1
      #X is the eigenvectors
      ind = eig_inds[i]
      eigenvector = U * onehot(ComplexF64, ind)
      psi[b] = eigenvector

      overlaps[ind.second] = abs(inner(psi, target))^2
      sum_overlap += overlaps[ind.second]
      if overlaps[ind.second] > overlaps[max_ind]
        max_ind = ind.second
      end

      if overlaps[max_ind] > (1 - sum_overlap)
        global max_overlap = overlaps[max_ind]
        # println("max_ind: ", max_ind)
        # println("max overlap: ", max_overlap)
        # println("N_iterations: ", N_iterations)
        break
      end
    end
  end

  u = uniqueind(U, H)

  U_max = 0
  eig = 0

  U_max = U * dag(onehot(u => max_ind))
  eig = D[max_ind, max_ind]

  return U_max, eig, spec, psi
end

function dmrgX(PH, psi0::MPS, targetPsi::MPS, sweeps::Sweeps; kwargs...)
  if length(psi0) == 1
    error(
      "`dmrg` currently does not support system sizes of 1. You can diagonalize the MPO tensor directly with tools like `LinearAlgebra.eigen`, `KrylovKit.eigsolve`, etc.",
    )
  end

  # @debug_check begin
  #   # Debug level checks
  #   # Enable with ITensors.enable_debug_checks()
  #   checkflux(psi0)
  #   checkflux(PH)
  # end
  n_eigs = get(kwargs, :n_eigs, 10)

  which_decomp::Union{String,Nothing} = get(kwargs, :which_decomp, "svd")
  svd_alg = get(kwargs, :svd_alg, nothing)#"divide_and_conquer")
  obs = get(kwargs, :observer, NoObserver())
  outputlevel::Int = get(kwargs, :outputlevel, 1)

  write_when_maxdim_exceeds::Union{Int,Nothing} = get(
    kwargs, :write_when_maxdim_exceeds, nothing
  )
  write_path = get(kwargs, :write_path, tempdir())
  is_gs = get(kwargs, :groundstate, false)

  # eigsolve kwargs
  eigsolve_tol::Number = get(kwargs, :eigsolve_tol, 1e-14)

  eigsolve_krylovdim::Int = get(kwargs, :eigsolve_krylovdim, n_eigs)
  eigsolve_maxiter::Int = get(kwargs, :eigsolve_maxiter, 1)
  eigsolve_verbosity::Int = get(kwargs, :eigsolve_verbosity, 0)

  ishermitian::Bool = get(kwargs, :ishermitian, true)
  limits::Vector = get(kwargs, :limits, [])

  gpu::Bool = get(kwargs, :gpu, false)
  exact_diag::Bool = get(kwargs, :exact_diag, true)

  if outputlevel >= 1
    println("Exact diagonalization: ", exact_diag)
    println("GPU usage: ", gpu)
    println("IS GROUND STATE: ", is_gs)
  end
  # TODO: add support for targeting other states with DMRG
  # (such as the state with the largest eigenvalue)
  # get(kwargs, :eigsolve_which_eigenvalue, :SR)
  eigsolve_which_eigenvalue::Symbol = :SR

  # TODO: use this as preferred syntax for passing arguments
  # to eigsolve
  #default_eigsolve_args = (tol = 1e-14, krylovdim = 3, maxiter = 1,
  #                         verbosity = 0, ishermitian = true,
  #                         which_eigenvalue = :SR)
  #eigsolve = get(kwargs, :eigsolve, default_eigsolve_args)

  # Keyword argument deprecations
  if haskey(kwargs, :maxiter)
    error("""maxiter keyword has been replaced by eigsolve_krylovdim.
             Note: compared to the C++ version of ITensor,
             setting eigsolve_krylovdim 3 is the same as setting
             a maxiter of 2.""")
  end

  if haskey(kwargs, :errgoal)
    error("errgoal keyword has been replaced by eigsolve_tol.")
  end

  if haskey(kwargs, :quiet)
    error("quiet keyword has been replaced by outputlevel")
  end

  psi = copy(psi0)
  N = length(psi)

  if !isortho(psi) || orthocenter(psi) != 1
    psi = orthogonalize!(PH, psi, 1)
  end
  if !isortho(targetPsi) || orthocenter(targetPsi) != 1
    targetPsi = orthogonalize!(PH, targetPsi, 1)
  end
  @assert isortho(psi) && orthocenter(psi) == 1

  if !isnothing(write_when_maxdim_exceeds)
    if (maxlinkdim(psi) > write_when_maxdim_exceeds) ||
      (maxdim(sweeps, 1) > write_when_maxdim_exceeds)
      PH = disk(PH; path=write_path)
    end
  end
  PH = position!(PH, psi, 1)
  energy = 0.0

  for sw in 1:nsweep(sweeps)
    sw_time = @elapsed begin
      maxtruncerr = 0.0

      if !isnothing(write_when_maxdim_exceeds) &&
        maxdim(sweeps, sw) > write_when_maxdim_exceeds
        if outputlevel >= 2
          println(
            "\nWriting environment tensors do disk (write_when_maxdim_exceeds = $write_when_maxdim_exceeds and maxdim(sweeps, sw) = $(maxdim(sweeps, sw))).\nFiles located at path=$write_path\n",
          )
        end
        PH = disk(PH; path=write_path)
      end

      for (b, ha) in sweepnext(N)

        #@timeit_debug timer "dmrg: position!" begin
        PH = position!(PH, psi, b)

        #@timeit_debug timer "dmrg: psi[b]*psi[b+1]" begin
        phi = psi[b] * psi[b + 1]
        # end
        targetPsi = orthogonalize!(targetPsi, b)
        targetphi = targetPsi[b] * targetPsi[b + 1]

        #println("Limits" , limits)
        ortho = ha == 1 ? "left" : "right"

        # CUDA.reclaim()

        (phi, energy, spec, psi) = dmrg_x_solver(
          PH,
          psi,
          targetPsi,
          b,
          sweeps,
          sw,
          ortho,
          which_decomp,
          svd_alg;
          limits=limits,
          obs=obs,
          is_gs=is_gs,
          exact_diag=exact_diag,
          phi=phi,
        )

        drho = nothing
        if noise(sweeps, sw) > 0.0
          #@timeit_debug timer "dmrg: noiseterm" begin
          # Use noise term when determining new MPS basis
          drho = noise(sweeps, sw) * noiseterm(PH, phi, ortho)
          #end
        end

        maxtruncerr = max(maxtruncerr, spec.truncerr)

        sweep_is_done = (b == 1 && ha == 2)
        measure!(
          obs;
          energy=energy,
          psi=psi,
          projected_operator=PH,
          bond=b,
          sweep=sw,
          half_sweep=ha,
          spec=spec,
          outputlevel=outputlevel,
          sweep_is_done=sweep_is_done,
        )
      end
    end
    if outputlevel >= 1
      @printf(
        "After sweep %d energy=%s  maxlinkdim=%d maxerr=%.2E time=%.3f\n",
        sw,
        energy,
        maxlinkdim(psi),
        maxtruncerr,
        sw_time
      )
      println("ENERGY: ", energy)
    end
    isdone = checkdone!(obs; energy=energy, psi=psi, sweep=sw, outputlevel=outputlevel)
    isdone && break
    # if gpu
    #   GC.gc()
    # end
  end
  # energy = real(inner(psi, apply(obs.H, psi; cutoff=0)))
  return (energy, psi)
end

function dmrgX_one_site(x1, psi0::MPS, targetPsi::MPS; kwargs...)
  return dmrgX_one_site(x1, psi0, targetPsi, _dmrg_sweeps(; kwargs...); kwargs...)
end

function dmrgX_one_site(x1, targetPsi::MPS; kwargs...)
  psi0 = deepcopy(targetPsi)
  return dmrgX_one_site(x1, psi0, targetPsi, _dmrg_sweeps(; kwargs...); kwargs...)
end

function dmrgX_one_site(H::MPO, psi0::MPS, targetPsi::MPS, sweeps::Sweeps; kwargs...)
  check_hascommoninds(siteinds, H, psi0)
  check_hascommoninds(siteinds, H, psi0')
  # Permute the indices to have a better memory layout
  # and minimize permutations
  H = permute(H, (linkind, siteinds, linkind))
  PH = ProjMPO(H)
  PH.nsite = 1
  return dmrgX_one_site(PH, psi0, targetPsi, sweeps; kwargs...)
end

function dmrgX_one_site(PH, psi0::MPS, targetPsi::MPS, sweeps::Sweeps; kwargs...)
  if length(psi0) == 1
    error(
      "`dmrg` currently does not support system sizes of 1. You can diagonalize the MPO tensor directly with tools like `LinearAlgebra.eigen`, `KrylovKit.eigsolve`, etc.",
    )
  end

  n_eigs = get(kwargs, :n_eigs, 10)
  is_gs = get(kwargs, :groundstate, false)
  which_decomp::Union{String,Nothing} = get(kwargs, :which_decomp, nothing)
  svd_alg::String = get(kwargs, :svd_alg, "divide_and_conquer")
  obs = get(kwargs, :observer, NoObserver())
  outputlevel::Int = get(kwargs, :outputlevel, 1)

  write_when_maxdim_exceeds::Union{Int,Nothing} = get(
    kwargs, :write_when_maxdim_exceeds, nothing
  )
  write_path = get(kwargs, :write_path, tempdir())
  # eigsolve kwargs
  eigsolve_tol::Number = get(kwargs, :eigsolve_tol, 1e-14)

  eigsolve_krylovdim::Int = get(kwargs, :eigsolve_krylovdim, n_eigs)
  eigsolve_maxiter::Int = get(kwargs, :eigsolve_maxiter, 1)
  eigsolve_verbosity::Int = get(kwargs, :eigsolve_verbosity, 0)

  ishermitian::Bool = get(kwargs, :ishermitian, true)

  gpu::Bool = get(kwargs, :gpu, false)
  exact_diag::Bool = get(kwargs, :exact_diag, true)
  # TODO: add support for targeting other states with DMRG
  # (such as the state with the largest eigenvalue)
  # get(kwargs, :eigsolve_which_eigenvalue, :SR)
  eigsolve_which_eigenvalue::Symbol = :SR

  if outputlevel >= 1
    println("Exact diagonalization: ", exact_diag)
    println("GPU usage: ", gpu)
    println("IS GROUND STATE: ", is_gs)
  end
  if haskey(kwargs, :maxiter)
    error("""maxiter keyword has been replaced by eigsolve_krylovdim.
             Note: compared to the C++ version of ITensor,
             setting eigsolve_krylovdim 3 is the same as setting
             a maxiter of 2.""")
  end

  if haskey(kwargs, :errgoal)
    error("errgoal keyword has been replaced by eigsolve_tol.")
  end

  if haskey(kwargs, :quiet)
    error("quiet keyword has been replaced by outputlevel")
  end

  psi = copy(psi0)
  N = length(psi)

  if !isortho(psi) || orthocenter(psi) != 1
    psi = orthogonalize!(PH, psi, 1)
  end
  if !isortho(targetPsi) || orthocenter(targetPsi) != 1
    targetPsi = orthogonalize!(PH, targetPsi, 1)
  end
  @assert isortho(psi) && orthocenter(psi) == 1

  if !isnothing(write_when_maxdim_exceeds)
    if (maxlinkdim(psi) > write_when_maxdim_exceeds) ||
      (maxdim(sweeps, 1) > write_when_maxdim_exceeds)
      PH = disk(PH; path=write_path)
    end
  end
  PH = position!(PH, psi, 1)
  energy = 0.0

  for sw in 1:nsweep(sweeps)
    sw_time = @elapsed begin
      maxtruncerr = 0.0

      if !isnothing(write_when_maxdim_exceeds) &&
        maxdim(sweeps, sw) > write_when_maxdim_exceeds
        if outputlevel >= 2
          println(
            "\nWriting environment tensors do disk (write_when_maxdim_exceeds = $write_when_maxdim_exceeds and maxdim(sweeps, sw) = $(maxdim(sweeps, sw))).\nFiles located at path=$write_path\n",
          )
        end
        PH = disk(PH; path=write_path)
      end

      for (b, ha) in sweepnext(N; ncenter=1)
        PH = position!(PH, psi, b)

        phi = psi[b]
        # end
        targetPsi = orthogonalize!(targetPsi, b)
        targetphi = targetPsi[b]

        ortho = ha == 1 ? "left" : "right"

        (phi, energy, spec, psi) = dmrg_x_solver_ss(
          PH,
          psi,
          targetPsi,
          b,
          sweeps,
          sw,
          ortho,
          which_decomp,
          svd_alg;
          obs=obs,
          is_gs=is_gs,
          exact_diag=exact_diag,
          phi=phi,
        )

        # maxtruncerr = max(maxtruncerr, spec.truncerr)

        sweep_is_done = (b == 1 && ha == 2)
        measure!(
          obs;
          energy=energy,
          psi=psi,
          projected_operator=PH,
          bond=b,
          sweep=sw,
          half_sweep=ha,
          spec=spec,
          outputlevel=outputlevel,
          sweep_is_done=sweep_is_done,
        )
      end
    end
    if outputlevel >= 1
      @printf(
        "After sweep %d energy=%s  maxlinkdim=%d time=%.3f\n",
        sw,
        energy,
        maxlinkdim(psi),
        sw_time
      )
      println("ENERGY: ", energy)
    end
    isdone = checkdone!(obs; energy=energy, psi=psi, sweep=sw, outputlevel=outputlevel)
    isdone && break
    # if gpu
    #   GC.gc()
    # end
  end
  # energy = real(inner(psi, apply(obs.H, psi; cutoff=0)))
  return (energy, psi)
end