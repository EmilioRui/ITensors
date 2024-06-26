using BlockArrays: AbstractBlockStyle
using Base.Broadcast: Broadcasted, flatten
struct BlockSparseStyle{N} <: AbstractBlockStyle{N} end
Base.BroadcastStyle(::Type{<:BlockSparseArray{<:Any,N}}) where {N} = BlockSparseStyle{N}()

BlockSparseStyle(::Val{N}) where {N} = BlockSparseStyle{N}()
BlockSparseStyle{M}(::Val{N}) where {N,M} = BlockSparseStyle{N}()

function Base.similar(bc::Broadcasted{BlockSparseStyle{N}}, ::Type{T}) where {T,N}
  return BlockSparseArray{T,N}(undef, axes(bc))
end

function preserves_zeros(f, elts::Type...)
  return iszero(f(zero.(elts)...))
end

function preserves_zeros(f, as::AbstractArray...)
  return preserves_zeros(f, eltype.(as)...)
end

function _broadcast(f, as::AbstractArray...)
  if !preserves_zeros(f, as...)
    error("Broadcasting functions that don't preserve zeros isn't supported yet.")
  end
  # TODO: Use `map_nonzeros` here?
  return map(f, as...)
end

function _broadcast!(f, a_dest::AbstractArray, as::AbstractArray...)
  if !preserves_zeros(f, as...)
    error("Broadcasting functions that don't preserve zeros isn't supported yet.")
  end
  # TODO: Use `map_nonzeros!` here?
  return map!(f, a_dest, as...)
end

isnumber(x::Number) = true
isnumber(x) = false

function flatten_numbers(f, args)
  # TODO: Is there a simpler way to implement this?
  # This doesn't play well with `Base.promote_op`.
  function flattened_f(flattened_args...)
    j = 0
    unflattened_args = ntuple(length(args)) do i
      if isnumber(args[i])
        return args[i]
      else
        return flattened_args[j += 1]
      end
    end
    return f(unflattened_args...)
  end
  return flattened_f, filter(!isnumber, args)
end

function Base.copy(bc::Broadcasted{<:BlockSparseStyle})
  bcf = flatten(bc)
  f, args = flatten_numbers(bcf.f, bcf.args)
  return _broadcast(f, args...)
end

function Base.copyto!(a_dest::BlockSparseArray, bc::Broadcasted{<:BlockSparseStyle})
  bcf = flatten(bc)
  f, args = flatten_numbers(bcf.f, bcf.args)
  return _broadcast!(f, a_dest, args...)
end

## # Special algebra cases
## struct LeftMul{C}
##   c::C
## end
## (f::LeftMul)(x) = f.c * x
## 
## struct RightMul{C}
##   c::C
## end
## (f::RightMul)(x) = x * f.c
## 
## # 2 .* a
## function Base.copy(bc::Broadcasted{<:BlockSparseStyle,<:Any,typeof(*),<:Tuple{<:Number,<:AbstractArray}})
##   # TODO: Use `map_nonzeros`.
##   return map(LeftMul(bc.args[1]), bc.args[2])
## end
## 
## # a .* 2
## function Base.copy(bc::Broadcasted{<:BlockSparseStyle,<:Any,typeof(*),<:Tuple{<:AbstractArray,<:Number}})
##   # TODO: Use `map_nonzeros`.
##   return map(RightMul(bc.args[2]), bc.args[1])
## end
## 
## # a ./ 2
## function Base.copy(bc::Broadcasted{<:BlockSparseStyle,<:Any,typeof(/),<:Tuple{<:AbstractArray,<:Number}})
##   # TODO: Use `map_nonzeros`.
##   return map(RightMul(inv(bc.args[2])), bc.args[1])
## end
## 
## # a .+ b
## function Base.copy(bc::Broadcasted{<:BlockSparseStyle,<:Any,<:Union{typeof(+),typeof(-)},<:Tuple{<:AbstractArray,<:AbstractArray}})
##   # TODO: Use `map_nonzeros`.
##   return map(+, bc.args...)
## end
