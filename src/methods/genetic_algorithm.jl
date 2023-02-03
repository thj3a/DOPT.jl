struct GeneticAlgorithm <: MetaHeuristic end

"""
"""

mutable struct Individual{T}
    #pos::Int            # current ant position
    sol::Vector{T}      # current solution
    obj::T              # objval value
    best_sol::Vector{T} # best solution ever
    best_obj::T         # best objval value ever
    #delta::Matrix{T}    # pheromone deposit

    Individual{T}(n::Integer) where {T} =
        #new(0, Vector{T}(undef, n), zero(T), Vector{T}(undef, n), zero(T), zeros(T, n, n))
        new(Vector{T}(undef, n), zero(T), Vector{T}(undef, n), zero(T))
end

function solve(
    method::GeneticAlgorithm,
    A::Matrix{T},
    ::Vector{K},
    s::Integer;
    params...,
) where {T,K<:Integer}
    return solve(method, A, s; params...)
end

"""
    solve(::AntColony, A::Matrix{T}, s::Integer; params...)

m : Number of 'experiments'
n : Length of each 'experiment'
s : Number of allowed 'experiments'
M : Number of ants
"""
function solve(
    ::GeneticAlgorithm,
    A::Matrix{T},
    s::Integer;
    max_iter::Union{Integer,Nothing} = 1_000,
    max_time::Union{Float64,Nothing} = 100.0,
    num_ants::Integer = size(A, 1),
    μ = 0.1, # mutation rate
    λ = 0.9, # crossover rate

    # α::Float64 = 1.0,
    # β::Float64 = 1.0,
    # ρ::Float64 = 0.5,
    # Q::Float64 = 1.0,
    params...
) where {T}
    m = size(A, 1)
    n = size(A, 2)
    p = Individual{T}[Individual{T}(m) for _ = 1:num_ants]
    τ = ones(T, m, m)
    η = zeros(T, m, m)

    num_iter  = 0
    run_time  = 0.0
    init_time = 0.0

    while !stop(num_iter, max_iter, run_time, max_time)
        # 1. Generate solutions
        #   1.1 Reset ant position and solution
        for 🧍 in p
            k = rand(1:m)

            🧍.pos = k # Reset position 
            🧍.sol[:] .= 1 # Reset solution 
            🧍.sol[k] = 0 # Set initial position as visited

            🧍.delta[:, :] .= 0 # Reset Pheromone deposits
        end

        #   1.3 Move ants around
        Threads.@threads for l = 1:num_ants
            🧍 = p[l]

            for _ = 1:s-1
                i = 🧍.pos
                γ = 🧍.sol' * @view(τ[i, :]) #.^ α  # + 🐜.sol' * η[i, :] .^ β # allowed transitions

                p = cumsum(🧍.sol .* @view(τ[i, :]) ./ γ)
                j = searchsortedfirst(p, rand()) # roulette

                🧍.pos    = j   # update current position
                🧍.sol[j] = 0.0 # mark as visited

                🧍.delta[i, j] += 1.0
            end

            🧍.obj = objval(A, (1.0 .- 🧍.sol))

            🧍.delta[:, :] *= Q / -🧍.obj

            # Update solutions
            if iszero(num_iter) || 🧍.obj > 🧍.best_obj
                🧍.best_obj = 🧍.obj
                🧍.best_sol[:] .= 🧍.sol[:]
            end
        end

        # Update pheromones
        for i = 1:m, j = 1:m
            τ[i, j] = (1 - ρ) * τ[i, j] + sum(🐜.delta[i, j] for 🐜 in p)
        end

        if iszero(num_iter)
            init_time = time()
        end

        run_time = time() - init_time
        num_iter += 1
    end

    x⃰, z⃰ = argmax(last, [((1.0 .- 🐜.best_sol), 🐜.best_obj) for 🐜 in p])

    return (x⃰, z⃰, num_iter, 0)
end

function mutate(x⃰::Vector{T}, μ::Float64) where {T}
    n = length(x⃰)
    for i = 1:n
        if rand() < μ
            x⃰[i] = rand(0:1)
        end
    end
    return x⃰
end

function crossover(x⃰::Vector{T}, y⃰::Vector{T}, λ::Float64) where {T}
    n = length(x⃰)
    for i = 1:n
        if rand() < λ
            x⃰[i], y⃰[i] = y⃰[i], x⃰[i]
        end
    end
    return x⃰, y⃰
end

function select(p::Vector{Individual{T}}) where {T}
    n = length(p)
    x⃰ = rand(1:n)
    y⃰ = rand(1:n)
    while x⃰ == y⃰
        y⃰ = rand(1:n)
    end
    return p[x⃰], p[y⃰]
end

function selection(p::Vector{Individual{T}}) where {T}
    n = length(p)
    q = Individual{T}[Individual{T}(n) for _ = 1:n]
    for i = 1:n
        x⃰, y⃰ = select(p)
        q[i].sol[:] .= x⃰.sol[:]
        q[i].obj = x⃰.obj
    end
    return q
end