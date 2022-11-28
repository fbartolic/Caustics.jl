module Utils

function match_points(a, b)
    idcs = zeros(Int, length(a)) 

    for i in 1:length(a)
        closest_idcs = sortperm(abs2.(a[i] .- b))

        # Store closest idx that is not already in idcs
        for j in 1:length(closest_idcs)
            if !(closest_idcs[j] in idcs)
                idcs[i] = closest_idcs[j]
                break
            end
        end
    end
    return idcs
end

end