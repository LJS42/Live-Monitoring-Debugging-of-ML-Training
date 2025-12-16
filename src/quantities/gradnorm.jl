struct GradNormQuantity <: AbstractQuantity
    key::Symbol
    GradNormQuantity() = new(:gradnorm)
end

quantity_key(q::GradNormQuantity) = q.key

_norm_sq(x::AbstractArray{<:Number}) = sum(abs2, x)
_norm_sq(x::Union{Tuple,NamedTuple}) = sum(_norm_sq, values(x))
_norm_sq(x::Nothing) = 0.0
_norm_sq(x) = 0.0

function compute!(q::GradNormQuantity, loss, grads)
    # grads is a nested NamedTuple structure
    # thus recursively sum the squared norms of all arrays in it
    return sqrt(_norm_sq(grads))
end
