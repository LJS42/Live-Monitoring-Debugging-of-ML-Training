using WGLMakie
using Bonito
using Statistics
using Dates
using Makie
using Base.Threads

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

    fps = 30
    quantity_data = Dict{Symbol,Vector{Point2f}}()

    Timer(1/fps, interval=1/fps) do t
        has_new_data = false

        while isready(channel)
            step, received_quantities = take!(channel)
            for (key, value) in received_quantities
                if haskey(quantity_data, key)
                    data = get!(quantity_data, key, Point2f[])
                    push!(data, Point2f(step, value))
                end
            end
            has_new_data = true
        end

        if has_new_data
            for (key, obs) in observables
                obs[] = get!(quantity_data, key, Point2f[])
            end
            for (key, ax) in axs
                autolimits!(ax)
            end
        end
    end
end
