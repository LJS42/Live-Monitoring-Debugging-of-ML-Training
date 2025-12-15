using WGLMakie
using Bonito
using Statistics
using Dates
using Makie
using Base.Threads

# coverage: ignore start
function setup_plots(quantities::Vector{<:AbstractQuantity})
    set_theme!(theme_black())
    fig = Figure(size=(1000, 600), fontsize=18)
    axs = Dict{Symbol,Axis}()
    observables = Dict{Symbol,Observable}()

    for (i, q) in enumerate(quantities)
        key = quantity_key(q)
        ax = Axis(fig[i, 1], title=string(key), xlabel="Step", ylabel="Value", yscale=log10)
        axs[key] = ax
        obs = Observable(Point2f[])
        lines!(ax, obs, color=:cyan)
        observables[key] = obs
    end

    WGLMakie.activate!()
    Bonito.browser_display()
    display(fig)

    return fig, observables, axs
end

function render_loop(
    channel::Channel,
    quantities::Vector{<:AbstractQuantity},
)
    fig, observables, axs = setup_plots(quantities)

    quantity_data = Dict{Symbol,Vector{Point2f}}()
    for (key, obs) in observables
        quantity_data[key] = obs[]
    end

    for (step, received_quantities) in channel
        for (key, value) in received_quantities
            if haskey(quantity_data, key)
                data = quantity_data[key]
                push!(data, Point2f(step, value))
            end
        end

        for (key, obs) in observables
            obs[] = quantity_data[key]
        end
        for (key, ax) in axs
            autolimits!(ax)
        end
    end
end
# coverage: ignore end