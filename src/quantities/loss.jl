struct LossQuantity <: AbstractQuantity
    key::Symbol # lightweight, immutable identifier, starts with : (e.g. :loss)
    LossQuantity() = new(:loss)
end

quantity_key(q::LossQuantity) = q.key

function compute!(q::LossQuantity, loss, grads)
    return loss
end
