
# macro timeit(args...)
#   # Redefine the behavior of the @timeit macro here
#   return quote end
# end

function islinkind(ind::Index)
  """
  returns if the index is a linkindex (if Link keyword is in the tags)
  """
  tags = ind.tags
  return "Link" in [String(tag) for tag in tags]
end

function directsum!(
  ψ::MPST, tensor::ITensor, position::Int, direction::Vector{}
) where {MPST<:AbstractMPS}
  n = length(ψ)

  t1 = ψ[position]
  tensors = [t1, tensor]
  Φ, new_idx = directsum(
    (tensors[i] => (direction[i],) for i in 1:length(tensors))...;
    tags=[tags(first(direction))],
  )

  ψ[position] = Φ
  return new_idx
end

function change_index(tensor::ITensor, old_index::Index, new_index::Index)
  @assert old_index.space == new_index.space
  new_inds = [ind == old_index ? new_index : ind for ind in inds(tensor)]
  array = Array(tensor, inds(tensor))
  return itensor(array, new_inds)
end

function dmrg_x_solver3S(
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
  N = length(psi0)
  alpha = noise(sweeps, sw)

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

  M_optim = psi[b]
  overlaps_time = @elapsed begin
    is_growing = abs(D[1, 1]) < abs(D[end, end])
    iteration_range = is_growing ? (1:(eig_inds.space)) : (eig_inds.space):-1:1
    N_iterations = 0
    spec = []
    # println("b= ", b)
    for i in iteration_range
      N_iterations += 1
      temp_psi = deepcopy(psi)
      #X is the eigenvectors
      ind = eig_inds[i]
      M_optim = U * onehot(ComplexF64, ind)

      #We now implement the supspace expansion
      ha = ortho == "left" ? 1 : -1

      #creating P
      temp_psi[b] = M_optim
      if (b == 1 && ha == 1) || (b == N && ha == -1)
        P = temp_psi[b] * PH.H[b]
      else
        P = PH.LR[b - ha] * temp_psi[b] * PH.H[b]
      end
      P *= alpha
      #position of the link is b for left to right sweep and b-1 for right to left
      linkpos = b + Int((ha - 1) / 2)
      #The new link on position linkpos is the reshaping of the linkdims of the MPO and MPS
      inds_to_reshape = [linkind(temp_psi, linkpos), linkind(PH.H, linkpos)]
      #reshaping and saving the new linkindex
      C = combiner(inds_to_reshape; tags=tags(first(inds_to_reshape)))
      new_index = uniqueind(C, P)
      P *= C
      noprime!(P)
      #selecting the direction to do dirsum ()
      direction = [linkind(temp_psi, linkpos), new_index]
      old_idx = direction[1]
      new_linkind = first(directsum!(temp_psi, P, b, direction))

      #adding zeros to the neighbouring site
      bd = new_index.space
      new_index = Index(bd)
      current_idxs = inds(temp_psi[b + ha])
      new_idcs = [ind == old_idx ? new_index : ind for ind in current_idxs]
      new_tensor = ITensor(0, new_idcs)
      direction = [old_idx, new_index]
      temp_link_ind = first(directsum!(temp_psi, new_tensor, b + ha, direction))
      new_idcs = [
        ind == temp_link_ind ? new_linkind : ind for ind in inds(temp_psi[b + ha])
      ]
      temp_psi[b + ha] = change_index(temp_psi[b + ha], temp_link_ind, new_linkind)

      if ortho == "left"
        L, R, spec = factorize(
          temp_psi[b],
          [linkind(temp_psi, b - 1), siteind(temp_psi, b)];
          maxdim=maxdim(sweeps, sw),
          mindim=mindim(sweeps, sw),
          cutoff=cutoff(sweeps, sw),
          ortho=ortho,
          which_decomp,
          svd_alg,
          tags=tags(linkind(temp_psi, linkpos)),
        )
        temp_psi[b] = L
        temp_psi[b + ha] *= R

      else
        L, R, spec = factorize(
          temp_psi[b],
          [linkind(temp_psi, b - 1)];
          maxdim=maxdim(sweeps, sw),
          mindim=mindim(sweeps, sw),
          cutoff=cutoff(sweeps, sw),
          ortho=ortho,
          which_decomp,
          svd_alg,
          tags=tags(linkind(psi, linkpos)),
        )
        temp_psi[b] = R
        temp_psi[b + ha] *= L
      end

      normalize!(temp_psi)

      overlaps[ind.second] = abs(inner(temp_psi, target))^2
      sum_overlap += overlaps[ind.second]
      if overlaps[ind.second] > overlaps[max_ind]
        max_ind = ind.second
      end
      if overlaps[max_ind] > (1 - sum_overlap)
        global max_overlap = overlaps[max_ind]
        psi = temp_psi
        break
      end
    end
  end
  #it is probably better with pure SVD as I am not sure where are the singular values
  u = uniqueind(U, H)

  U_max = 0
  eig = 0

  U_max = U * dag(onehot(u => max_ind))
  eig = D[max_ind, max_ind]

  return U_max, eig, spec, psi
end

function dmrgX3S(x1, psi0::MPS, targetPsi::MPS; kwargs...)
  return dmrgX3S(x1, psi0, targetPsi, _dmrg_sweeps(; kwargs...); kwargs...)
end

function dmrgX3S(x1, targetPsi::MPS; kwargs...)
  psi0 = deepcopy(targetPsi)
  return dmrgX3S(x1, psi0, targetPsi, _dmrg_sweeps(; kwargs...); kwargs...)
end

function dmrgX3S(H::MPO, psi0::MPS, targetPsi::MPS, sweeps::Sweeps; kwargs...)
  check_hascommoninds(siteinds, H, psi0)
  check_hascommoninds(siteinds, H, psi0')
  # Permute the indices to have a better memory layout
  # and minimize permutations
  H = permute(H, (linkind, siteinds, linkind))
  PH = ProjMPO(H)
  PH.nsite = 1
  return dmrgX3S(PH, psi0, targetPsi, sweeps; kwargs...)
end

function dmrgX3S(PH, psi0::MPS, targetPsi::MPS, sweeps::Sweeps; kwargs...)
  if length(psi0) == 1
    error(
      "`dmrg` currently does not support system sizes of 1. You can diagonalize the MPO tensor directly with tools like `LinearAlgebra.eigen`, `KrylovKit.eigsolve`, etc.",
    )
  end

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

  eigsolve_krylovdim::Int = get(kwargs, :eigsolve_krylovdim, 2)
  eigsolve_maxiter::Int = get(kwargs, :eigsolve_maxiter, 1)
  eigsolve_verbosity::Int = get(kwargs, :eigsolve_verbosity, 0)

  ishermitian::Bool = get(kwargs, :ishermitian, true)

  gpu::Bool = get(kwargs, :gpu, false)
  exact_diag::Bool = get(kwargs, :exact_diag, true)
  # TODO: add support for targeting other states with DMRG
  # (such as the state with the largest eigenvalue)
  # get(kwargs, :eigsolve_which_eigenvalue, :SR)

  if outputlevel >= 1
    println("Exact diagonalization: ", exact_diag)
    println("GPU usage: ", gpu)
    println("IS GROUND STATE: ", is_gs)
    println("SINGLE SITEEEEE")
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

  psi = deepcopy(psi0)
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
        if (b == N && ha == 1) || (b == 1 && ha == 2)
          continue
        end
        PH = position!(PH, psi, b)

        phi = psi[b]
        # end
        targetPsi = orthogonalize!(targetPsi, b)
        targetphi = targetPsi[b]

        ortho = ha == 1 ? "left" : "right"

        (phi, energy, spec, psi) = dmrg_x_solver3S(
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


""" 
Single site dmrg X

"""
function dmrg_x_single_site_solver(
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
  gpu = occursin("CUDA",  string(typeof(psi0[1].tensor)))

  N = length(psi0)
  alpha = noise(sweeps, sw)

  H = contract(PH, ITensor(true))
  H⁺ = swapprime(dag(H), 0 => 1)
  # println(inds(H))
  time_diag = @elapsed begin
    D, U = eigen(.5*(H+H⁺); ishermitian=true)
  end

  U_inds = inds(U)[1:(end - 1)]
  eig_inds = inds(U)[end]
  psi = deepcopy(psi0)
  overlaps = zeros(eig_inds.space)
  # println("EIG INDS: ", eig_inds)
  sum_overlap = 0.0
  max_ind = eig_inds.space

  M_optim = psi[b]
  overlaps_time = @elapsed begin
    is_growing = real(D[1, 1]) < real(D[end, end])
    # println(is_growing)
    # iteration_range = is_growing ? (1:(eig_inds.space)) : (eig_inds.space):-1:1
    iteration_range = (1:(eig_inds.space))
    N_iterations = 0
    spec = []
    # println("b= ", b)
    for i in iteration_range
      N_iterations += 1
      temp_psi = deepcopy(psi)
      #X is the eigenvectors
      ind = eig_inds[i]
      if gpu
        M_optim = U * NDTensors.cu(onehot(ComplexF64, ind))
      else
        M_optim = U * onehot(ComplexF64, ind)
      end

      #We now implement the supspace expansion
      ha = ortho == "left" ? 1 : -1

      #creating P
      temp_psi[b] = M_optim
    
      normalize!(temp_psi)

      overlaps[ind.second] = abs(inner(temp_psi, target))^2
      sum_overlap += overlaps[ind.second]
      if overlaps[ind.second] > overlaps[max_ind]
        max_ind = ind.second
      end
      if overlaps[max_ind] > (1 - sum_overlap)
        global max_overlap = overlaps[max_ind]
        psi = deepcopy(temp_psi)
          println("max_ind: ", max_ind)
          println("max overlap: ", max_overlap)
          println("N_iterations: ", N_iterations)
        break
      end
    end
  end
  #it is probably better with pure SVD as I am not sure where are the singular values
  u = uniqueind(U, H)

  U_max = 0
  eig = 0

  if gpu
    U_max = U * NDTensors.cu(onehot(u => max_ind))
  else
    U_max = U * dag(onehot(u => max_ind))
  end
  eig = D[max_ind, max_ind]

  return U_max, eig, spec, psi
end

function dmrgX_single_site(x1, psi0::MPS, targetPsi::MPS; kwargs...)
  return dmrgX_single_site(x1, psi0, targetPsi, _dmrg_sweeps(; kwargs...); kwargs...)
end

function dmrgX_single_site(x1, targetPsi::MPS; kwargs...)
  psi0 = deepcopy(targetPsi)
  return dmrgX_single_site(x1, psi0, targetPsi, _dmrg_sweeps(; kwargs...); kwargs...)
end

function dmrgX_single_site(H::MPO, psi0::MPS, targetPsi::MPS, sweeps::Sweeps; kwargs...)
  check_hascommoninds(siteinds, H, psi0)
  check_hascommoninds(siteinds, H, psi0')
  # Permute the indices to have a better memory layout
  # and minimize permutations
  H = permute(H, (linkind, siteinds, linkind))
  PH = ProjMPO(H)
  PH.nsite = 1
  return dmrgX_single_site(PH, psi0, targetPsi, sweeps; kwargs...)
end

function dmrgX_single_site(PH, psi0::MPS, targetPsi::MPS, sweeps::Sweeps; kwargs...)
  if length(psi0) == 1
    error(
      "`dmrg` currently does not support system sizes of 1. You can diagonalize the MPO tensor directly with tools like `LinearAlgebra.eigen`, `KrylovKit.eigsolve`, etc.",
    )
  end

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

  eigsolve_krylovdim::Int = get(kwargs, :eigsolve_krylovdim, 2)
  eigsolve_maxiter::Int = get(kwargs, :eigsolve_maxiter, 1)
  eigsolve_verbosity::Int = get(kwargs, :eigsolve_verbosity, 0)

  ishermitian::Bool = get(kwargs, :ishermitian, true)

  gpu::Bool = get(kwargs, :gpu, false)
  exact_diag::Bool = get(kwargs, :exact_diag, true)
  # TODO: add support for targeting other states with DMRG
  # (such as the state with the largest eigenvalue)
  # get(kwargs, :eigsolve_which_eigenvalue, :SR)

  if outputlevel >= 1
    println("Exact diagonalization: ", exact_diag)
    println("GPU usage: ", gpu)
    println("IS GROUND STATE: ", is_gs)
    println("SINGLE SITEEEEE")
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

  psi = deepcopy(psi0)
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
        if (b == N && ha == 1) || (b == 1 && ha == 2)
          continue
        end
        PH = position!(PH, psi, b)

        phi = psi[b]
        # end
        targetPsi = orthogonalize!(targetPsi, b)
        targetphi = targetPsi[b]

        ortho = ha == 1 ? "left" : "right"

        (phi, energy, spec, psi) = dmrg_x_single_site_solver(
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