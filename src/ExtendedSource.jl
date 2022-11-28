module ExtendedSource

export mag_extended_source 

include("PointSource.jl")
include("Integrate.jl")
include("Utils.jl")

import .PointSource: images_point_source
import .Integrate:  integrate_unif, integrate_ld
import .Utils: match_points

mutable struct Segment 
    z::Vector{ComplexF64}
    parity::Int
end

function get_segment_length(segment::Segment)
    res = 0.
    for i in 1:length(segment.z) - 1
        res += abs(segment.z[i+1] - segment.z[i])
    end
    res += abs(segment.z[1] - segment.z[end])
    return res 
end

"""
Sequentially permute the images corresponding to points on the source limb
starting with the first point such that each point source image is assigned
to the correct curve. This procedure does not differentiate between real
and false images, false images are set to zero after the permutation
operation.
"""
function permute_images(z::Matrix{ComplexF64}, z_mask::Matrix{Bool}, z_parity::Matrix{Int})
    for i in 1:length(z[:, 1]) - 1
        idcs = match_points(z[i, :], z[i+1, :])
        z[i+1, :] = z[i+1, idcs]
        z_mask[i+1, :] = z_mask[i+1, idcs]
        z_parity[i+1, :] = z_parity[i+1, idcs]
    end
    return z, z_mask, z_parity
end

"""
Evaluate the point source images for a uniformly distributed set of `npts` points on the 
limb of the source disc with radius `rho`, located at `w0`. The output arrays have 
shape (npts, nimages).
"""
function limb_sampling_initial(
    w0::ComplexF64,
    ρ::Float64,
    params::Dict,
    nlenses=2::Int,
    npts=100::Int,
)
    θ = collect(LinRange(-π, π - 1e-08, npts))
    w_points = w0 .+ ρ.* exp.(1im .* θ)

    z =  zeros(ComplexF64, npts, nlenses^2 + 1)
    z_mask = zeros(Bool, npts, nlenses^2 + 1)
    z_parity = zeros(Int, npts, nlenses^2 + 1)

    for i in 1:npts
        if  i == 1
            z[i, :], z_mask[i, :], z_parity[i, :] = images_point_source(w_points[i], params, nlenses)
        else
            z[i, :], z_mask[i, :], z_parity[i, :] = images_point_source(w_points[i], params, nlenses, z[i-1, :])
        end
    end
    return θ, z, z_mask, z_parity 
end

"""
Given a set of images returnd by `limb_sampling_initial`, this function adds additional
`npts_add` images evaluated at points on the limb of the source disc where the gap 
between the images in the image plane is largest. 
"""
function limb_sampling_refine(
    w0::ComplexF64, 
    ρ::Float64, 
    params::Dict, 
    θ::Vector{Float64}, 
    z::Matrix{ComplexF64},
    z_mask::Matrix{Bool},
    z_parity::Matrix{Int},
    npts_add=20::Int, 
    nlenses=2::Int
)
    Δz = abs2.(z[2:end, :] .- z[1:end-1, :])  
    Δz = ifelse.(z_mask[2:end, :] .| z_mask[1:end-1, :], Δz, zeros(Float64, size(Δz)))
    Δz_max = reshape(maximum(Δz, dims=2), :)
    idcs_θ = sortperm(Δz_max, rev=true)[1:npts_add]
  
    # Evaluate images at new points
    θ_new = 0.5 * (θ[idcs_θ] + θ[idcs_θ .+ 1])
    w_points_new = w0 .+ ρ.* exp.(1im .* θ_new)

    # Add new points at the midpoints of the top-ranking intervals
    z_new = zeros(ComplexF64, npts_add, size(z, 2))
    z_mask_new = zeros(Bool, npts_add, size(z, 2))
    z_parity_new = zeros(Int, npts_add, size(z, 2))

    for i in 1:length(θ_new)
        z_new[i, :], z_mask_new[i, :], z_parity_new[i, :]= images_point_source(
            w_points_new[i], params, nlenses, z[idcs_θ[i], :]
        )
    end

    # Append new points to the end of existing arrays
    θ = vcat(θ, θ_new)
    z = vcat(z, z_new)
    z_mask = vcat(z_mask, z_mask_new)
    z_parity = vcat(z_parity, z_parity_new)

    # Sort 
    idcs_sort = sortperm(θ)
    θ = θ[idcs_sort]
    z = z[idcs_sort, :]
    z_mask = z_mask[idcs_sort, :]
    z_parity = z_parity[idcs_sort, :]

    return θ, z, z_mask, z_parity 
end

"""
Split a fixed-size array of point source images into variable sized chunks (segments) 
such that each part consists of only real images with the same parity and there are 
no large jumps in distance between consecutive images. 
"""
function process_open_segments(segments)
    # Split a single segment such that each part consists of only real images 
    # with the same parity such that there are no large jumps in distance between
    # consecutive images.
    segments_split = []

    # Iterate over open segments
    for seg in segments
        z, z_mask, z_parity = seg["z"], seg["z_mask"], seg["z_parity"]
        # Iterate over points in each segment
        idx = 1
        for i in 1:length(z) - 1
            if (z_mask[i] != z_mask[i+1]) || (z_parity[i] != z_parity[i+1]) || (i == length(z) - 1)
                # Split segment
                push!(segments_split, Dict("z" => z[idx:i], "z_mask" => z_mask[idx:i], "z_parity" => z_parity[idx:i]))
                idx = i + 1
            end
        end
        if idx != length(z) - 1
            push!(segments_split, Dict("z" => z[idx:end], "z_mask" => z_mask[idx:end], "z_parity" => z_parity[idx:end]))
        end
    end

    # Remove segments with false images or those with fewer than 3 points
    segments_final = []
    for seg in segments_split
        z, z_mask, z_parity = seg["z"], seg["z_mask"], seg["z_parity"]
        if (sum(z_mask) > 0) && length(z) >= 3
            push!(segments_final, Segment(z, z_parity[1]))
        end
    end
    return segments_final
end

"""
Given the raw images corresponding to a sequence of points on the source
limb, return two arrays with `open` and `closed` contour segments.
Closed segments are contours which do not cross the critical curve
and do not require any extra processing. Open segments need to be stitched 
together to form closed contours.
"""
function get_segments(z::Matrix{ComplexF64}, z_mask::Matrix{Bool}, z_parity::Matrix{Int})
    segments_interim_closed = []
    segments_interim_open = []

    # Split segments between open and closed
    for i in 1:length(z[1, :])
        if (abs2(z[1, i] - z[end, i]) < 1e-5^2) && all(z_mask[:, i])
            push!(segments_interim_closed, Dict("z" => z[:, i], "z_mask" => z_mask[:, i], "z_parity" => z_parity[:, i]))
        else
            push!(segments_interim_open, Dict("z" => z[:, i], "z_mask" => z_mask[:, i], "z_parity" => z_parity[:, i]))
        end
    end

    # Process closed segments
    segments_closed = Segment[]
    for seg in segments_interim_closed
        push!(segments_closed, Segment(seg["z"], seg["z_parity"][1]))
    end

    # Process open segments if there are any
    if length(segments_interim_open) > 0
        segments_open = process_open_segments(segments_interim_open)
    else
        segments_open = Segment[]
    end

    # TODO: check that there are no identical points at the ends of segments
    return segments_closed, segments_open
end

function merge_two_segments(seg1::Segment, seg2::Segment, ctype::String)
    if ctype == "hh"
        seg2.z = reverse(seg2.z) # flip
        seg2.parity = -seg2.parity # flip parity
        return Segment(vcat(seg2.z, seg1.z), seg2.parity)
    elseif ctype == "tt"
        seg2.z = reverse(seg2.z) # flip
        seg2.parity = -seg2.parity # flip parity
        return Segment(vcat(seg1.z, seg2.z), seg2.parity)
    elseif ctype == "th"
        return Segment(vcat(seg1.z, seg2.z), seg2.parity)
    elseif ctype == "ht"
        return Segment(vcat(seg2.z, seg1.z), seg2.parity)
    end
end

"""
Dermine wether two segments should be connected or not for a specific
type of connection. We differentiate between four types of connections:
    - `ctype` == 0: Tail-Head connection
    - `ctype` == 1: Head-Tail connection
    - `ctype` == 2: Head-Head connection
    - `ctype` == 3: Tail-Tail connection

We use four criterions to determine if the segments should be connected:
    1. For T-H and H-T the two segments need to have the same parity and
        for H-H and T-T connections they need to have opposite parity.
    2. If we form a line consisting of two points at the end of each segment,
        such that the second point is the connection point, the distance
        between two potential connection points of the segments must be
        less than the distance between the other two points.
    3. The smaller of the two angles formed by the intersection of `line1`
        and `line2` must be less than `max_ang` degrees.
    4. The distance between two potential connection points must be less
        than `max_dist`.

If the distance between the two connection points is less than `min_dist`,
and the parity condition is satisfied the function returns `True`
irrespective of the other conditions.

All of this is to ensure that we avoid connecting two segments which
shouldn't be connected.
"""
function connection_condition(
    seg1::Segment,
    seg2::Segment,
    ctype::String,
    min_dist=1e-05::Float64,
    max_dist=1e-01::Float64,
    max_ang=60.0::Float64,
)
    if ctype == "th"
        line1 = [seg1.z[end-1], seg1.z[end]]
        line2 = [seg2.z[2], seg2.z[1]]
    elseif ctype == "ht"
        line1 = [seg1.z[2], seg1.z[1]]
        line2 = [seg2.z[end-1], seg2.z[end]]
    elseif ctype == "hh"
        line1 = [seg1.z[2], seg1.z[1]]
        line2 = [seg2.z[2], seg2.z[1]]
    elseif ctype == "tt"
        line1 = [seg1.z[end-1], seg1.z[end]]
        line2 = [seg2.z[end-1], seg2.z[end]]
    end

    # If the distance between the two connection points is less than `min_dist`,
    # return true 
    if abs2(line1[2] - line2[2]) < min_dist^2
        return true
    end

    # Criterion 1
    if ctype == "th" || ctype == "ht"
        if seg1.parity != seg2.parity
            return false
        end
    elseif ctype == "hh" || ctype == "tt"
        if seg1.parity == seg2.parity
            return false
        end
    end

    # Criterion 2
    if abs2(line1[2] - line2[2]) > abs2(line1[1] - line2[1])
        return false
    end

    # Criterion 3
    vec1 = (line1[2] - line1[1]) / abs(line1[2] - line1[1])
    vec2 = (line2[2] - line2[1]) / abs(line2[2] - line2[1])
    α = acos(vec1.re.*vec2.re + vec1.im.*vec2.im) 
    if (180 - rad2deg(α)) > max_ang
        return false
    end

    # Criterion 4
    if abs2(line1[2] - line2[2]) > max_dist^2
        return false
    end

    return true
end


"""
Sequentially merge open segments until they form a closed contour. The
merging algorithm is as follows:

    1. Select shortest (in length) open segment, set it as the active segment.
    2. Find segment closest in distance to the active segment which satisfies 
        the connection condition. Merge active segment with that segment.
    3. Repeat step 2. until no more segments can be merged. If there are no 
        more segments to merge, the active segment is added to the list of 
        closed segments and the process terminates. If there are open 
        segments left, start again from step 1.
"""
function merge_open_segments(segments)
    # Sort segments by length from smallest to largest
    segments = sort(segments, by=seg -> get_segment_length(seg))
    segments_closed = Segment[] # Merged segments 

    function get_connection_distance(seg1, seg2, ctype)
        if ctype == "th"
            return abs(seg1.z[end] - seg2.z[1])
        elseif ctype == "ht"
            return abs(seg1.z[1] - seg2.z[end])
        elseif ctype == "hh"
            return abs(seg1.z[1] - seg2.z[1])
        elseif ctype == "tt"
            return abs(seg1.z[end] - seg2.z[end])
        end
    end

    segment_active = segments[1]
    deleteat!(segments, 1)

    while true
        # Search over all segments and connection types to find a segment to 
        # connect to (closest distance and valid connection)
        min_dist = Inf
        min_dist_index = 0
        min_dist_ctype = ""

        for i in 1:length(segments)
            seg = segments[i]
            for ctype in ["th", "ht", "hh", "tt"]
                if connection_condition(segment_active, seg, ctype)
                    dist = get_connection_distance(segment_active, seg, ctype)
                    if dist < min_dist
                        min_dist = dist
                        min_dist_index = i
                        min_dist_ctype = ctype
                    end
                end
            end
        end

        # If we found a segment to connect to, merge the two segments
        if min_dist_index != 0
            segment_active = merge_two_segments(segment_active, segments[min_dist_index], min_dist_ctype)
            deleteat!(segments, min_dist_index)

        # Otherwise save the active segment to merged segments and select a new one or exit the loop
        else
            # Save the active segment 
            push!(segments_closed, segment_active)

            if length(segments) > 0
                # Start a new active segment
                segments = sort(segments, by=seg -> get_segment_length(seg))
                segment_active = segments[1]
                deleteat!(segments, 1)
            else
                # No more segments to merge
                return segments_closed
            end
        end
    end
end

"""
Compute the magnification of an extended source with radius `rho` for a 
system with `nlenses` lenses. If `nlenses` is 2 (binary lens) or 3 
(triple lens), the coordinate system is set up such that the the origin is 
at the center of mass of the first two lenses which are both located on the 
real line. The location of the first lens is `-sq/(1 + q)`` and the second lens 
is at `s/(1 + q)``. The optional third lens is located at an arbitrary position 
in the complex plane `r_3e^{-i*psi}`. The magnification is computed using 
contour integration in the image plane. Boolean flag `limb_darkening` 
indicated whether linear limb-darkening needs to taken into account. If 
`limb_darkening` is set to True the linear limb-darkening coefficient 
u1 needs to be specified as well. 

If `nlenses` is 2 only the parameters `s` and `q` should be specified. If 
`nlenses` is 3, the parameters `s`, `q`, `q3`, `r3` and `psi` should be 
specified.
"""
function mag_extended_source(
    w0::ComplexF64, 
    ρ::Float64, 
    params::Dict, 
    nlenses=2::Int,
    u1=0.::Float64, 
    rtol=1e-03::Float64, 
    npts_init=100::Int, 
    npts_add=20::Int,  
    maxiter=10::Int
)
    if npts_init < 100
        println("WARNING: npts_init < 100. This may cause issues with the contour construction algorithm.")
    end

    s, q = params["s"], params["q"]
    a, e1 = 0.5*s, q/(1 + q)
    _params = Dict("a" => a, "e1" => e1)
    x_cm = 0.5*s*(1-q)/(1 + q)
    w0 -= x_cm

    # Initial sampling: uniformly sample `npts_init` points on the limb and then 
    # add additional `npts_init` points in 10 refinement steps
    θ, z, z_mask, z_parity = limb_sampling_initial(w0, ρ, _params, nlenses, npts_init)
    for _ in 1:10
        θ, z, z_mask, z_parity = limb_sampling_refine(w0, ρ, _params, θ, z, z_mask, z_parity, Int(round(npts_init/10.)))
    end

    # Final sampling: Add `npts_add` in a loop until the relative change in the
    # magnification is less than `rtol`
    err_rel = Inf
    mag = Inf
    contours = []
    
    i = 0
    while (err_rel > rtol) && (i < maxiter)
        # Refine sampling
        θ, z, z_mask, z_parity = limb_sampling_refine(w0, ρ, _params, θ, z, z_mask, z_parity, npts_add)

        # Permute image in correct order
        z, z_mask, z_parity = permute_images(z, z_mask, z_parity)

        # Get segments
        segments_closed, segments_open = get_segments(z, z_mask, z_parity)

        if length(segments_open) > 0
            segments_merged = merge_open_segments(segments_open)
        else
            segments_merged = []
        end

        # Concatenate closed and open segments
        contours = vcat(segments_closed, segments_merged)

        # Append first point in each contour to the end of the contour
        for contour in contours
            contour.z = vcat(contour.z, contour.z[1])
        end

        # Integrate 
        mag_new = 0.
        if u1 == 0. # uniform brightness source
            for contour in contours
                mag_new += integrate_unif(contour)*contour.parity/(2π*ρ^2)
            end
        else # limb-darkened source
            for contour in contours
                mag_new += integrate_ld(contour, w0, ρ, _params, u1, nlenses, rtol)*contour.parity/(2π*ρ^2)
            end
        end

        err_rel = abs((mag - mag_new)/mag_new)
        mag = mag_new
        i += 1
    end
    return i, contours, abs(mag)
end
end
