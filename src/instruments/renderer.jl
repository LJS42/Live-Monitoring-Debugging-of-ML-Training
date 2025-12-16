using WGLMakie
using Bonito
using Statistics
using Dates
using Base.Threads


"""
    setup_plots(quantities)  -> fig, observables, axs

Create a WGLMakie figure with one axis per tracked quantity.

For each quantity this function creates:
- an  axis at row i, column 1 of the figure grid
- an observable that holds the line data (step, value)
- a line plot that updates automatically when the observable is updated

The y-axis is displayed on a log10 scale to support values spanning orders
of magnitude (e.g. losses).

Args
    quantities: Vector{<:AbstractQuantity}
        objects defining quantities/metrics to plot

Returns
    fig: Figure
        the Makie figure
    observables: Dict{Symbol,Observable}
        mapping quantity key → observable line data, update of an observable results in plot update
    axs :Dict{Symbol,Axis}
        mapping quantity key → axis, adjust limits while plotting

"""
function setup_plots(
    quantities::Vector{<:AbstractQuantity};
    display::Bool = true,
)
    set_theme!(theme_black())

    fig = Figure(size=(1000, 600), fontsize=18)
    axs = Dict{Symbol,Axis}() 
    observables = Dict{Symbol,Observable}() 

    for (i, q) in enumerate(quantities)
        key = quantity_key(q)
        ax = Axis(
            fig[i, 1],
            title = string(key),
            xlabel = "Step",
            ylabel = "Value",
            yscale = log10,
        )
        axs[key] = ax
        obs = Observable(Point2f[])
        lines!(ax, obs, color = :cyan)
        observables[key] = obs
    end

    if display && !haskey(ENV, "CI")
        WGLMakie.activate!()
        Bonito.browser_display()
        display(fig)
    end
    return fig, observables, axs 
end

"""
    render_loop(channel, quantities)

Consume training updates from `channel` and update WGLMakie plots in real time.

Creates a figure using setup_plot and intialize observables data,

Iterates over messages from channel, expected to be (step, dict) where dict contains quantity values for that step.
Appends (step, value) points to the corresponding series and updates Makie observables to trigger plot redraws 
(use autolimits! to keep axes scaled to the data).

The loop terminates automatically when channel is closed and all messages have been passed.

Args:
    channel: Channel:
        stream of training updates (produced by training loop)
    quantities: Vector{<:AbstractQuantity}
        quantities defining which series to plot

"""
function render_loop(
    channel::Channel,
    quantities::Vector{<:AbstractQuantity};
    display::Bool = true,
)
    fig, observables, axs = setup_plots(quantities; display = display)

    quantity_data = Dict{Symbol,Vector{Point2f}}(
        key => copy(obs[]) for (key, obs) in observables
    )

    for (step, received_quantities) in channel #iterates over channel messages
    #blocks when channel is empty
        for (key, value) in received_quantities
            haskey(quantity_data, key) || continue
            push!(quantity_data[key], Point2f(step, value))
        end

        for (key, obs) in observables
            obs[] = quantity_data[key] #updated observables vectors 
        end
        for ax in values(axs)
            autolimits!(ax)
        end
    end
end
