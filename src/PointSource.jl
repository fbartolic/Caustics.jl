module PointSource

include("Utils.jl")

import .Utils: match_points

export images_point_source, critical_and_caustic_curves, mag_point_source

using StaticArrays
using PolynomialRoots

"""
poly_coeffs_binary(w::ComplexF64, a::Float64, e1::Float64)

Compute the coefficients of the complex polynomial equation corresponding
to the binary lens equation. The function returns a vector of coefficients
starting with the highest order term.

### Input

- `w::ComplexF64` -- Position in the source plane.
- `a::Float64` -- Half the separation between the two lenses. We use the
        convention where both lenses are located on the real line with
        r1 = a and r2 = -a.
- `e1::Float64` -- Mass fraction of the first lens `\epsilon_1 = m_1/(m_1 + m_2)`. It
        follows that `\epsilon_2 = 1 - \epsilon_1`.

### Output

Polynomial coefficients, in order of increasing power.
"""
function poly_coeffs_binary(w::ComplexF64, a::Float64, e1::Float64)
    wbar = conj(w)

    p_0 =-a^2 + wbar^2
    p_1 = a^2 * w - 2 * a * e1 + a - w * wbar^2 + wbar
    p_2 = (
        2 * a^4
        - 2 * a^2 * wbar^2
        + 4 * a * wbar * e1
        - 2 * a * wbar
        - 2 * w * wbar
    )
    p_3 = (
        -2 * a^4 * w
        + 4 * a^3 * e1
        - 2 * a^3
        + 2 * a^2 * w * wbar^2
        - 4 * a * w * wbar * e1
        + 2 * a * w * wbar
        + 2 * a * e1
        - a
        - w
    )
    p_4 = (
        -(a^6)
        + a^4 * wbar^2
        - 4 * a^3 * wbar * e1
        + 2 * a^3 * wbar
        + 2 * a^2 * w * wbar
        + 4 * a^2 * e1^2
        - 4 * a^2 * e1
        + 2 * a^2
        - 4 * a * w * e1
        + 2 * a * w
    )
    p_5 = (
        a^6 * w
        - 2 * a^5 * e1
        + a^5
        - a^4 * w * wbar^2
        - a^4 * wbar
        + 4 * a^3 * w * wbar * e1
        - 2 * a^3 * w * wbar
        + 2 * a^3 * e1
        - a^3
        - 4 * a^2 * w * e1^2
        + 4 * a^2 * w * e1
        - a^2 * w
    )

    p = SVector{6, ComplexF64}(p_5, p_4, p_3, p_2, p_1, p_0)

    return p
end


"""
poly_coeffs_triple(w::ComplexF64, a::Float64, r3::ComplexF64, e1::Float64, e2::Float64)

Compute the coefficients of the complex polynomial equation corresponding
to the triple lens equation. The function returns a vector of coefficients
starting with the highest order term.

### Input

- `w::ComplexF64` -- Position in the source plane.
- `a::Float64` -- Half the separation between the first two lenses located on
        the real line with `r_1 = a` and `r_2 = -a`.
- `r3::ComplexF64` -- The position of the third lens.
- `e1::Float64` -- Mass fraction of the first lens `\epsilon_1 = m_1/(m_1 + m_2 + m_3)`.
- `e2::Float64` -- Mass fraction of the second lens `\epsilon_2 = m_2/(m_1 + m_2 + m_3)`.

### Output

Polynomial coefficients, in order of increasing power.
"""


function poly_coeffs_triple(w::ComplexF64, a::Float64, r3::ComplexF64, e1::Float64, e2::Float64)
    wbar = conj(w)
    r3bar = conj(r3)

    p_0 = -(a^2) * wbar + a^2 * r3bar + wbar^3 - wbar^2 * r3bar

    p_1 = (
        a^2 * w * wbar
        - a^2 * w * r3bar
        + 3 * a^2 * wbar * r3
        - 3 * a^2 * r3bar * r3
        - a^2 * e1
        - a^2 * e2
        - a * wbar * e1
        + a * wbar * e2
        + a * r3bar * e1
        - a * r3bar * e2
        - w * wbar^3
        + w * wbar^2 * r3bar
        - 3 * wbar^3 * r3
        + 3 * wbar^2 * r3bar * r3
        + 2 * wbar^2
        + wbar * r3bar * e1
        + wbar * r3bar * e2
        - 2 * wbar * r3bar
    )

    p_2 = (
        3 * a^4 * wbar
        - 3 * a^4 * r3bar
        - a^3 * e1
        + a^3 * e2
        - 3 * a^2 * w * wbar * r3
        + 3 * a^2 * w * r3bar * r3
        + a^2 * w
        - 3 * a^2 * wbar^3
        + 3 * a^2 * wbar^2 * r3bar
        - 3 * a^2 * wbar * r3^2
        + 3 * a^2 * r3bar * r3^2
        + 4 * a^2 * e1 * r3
        + 4 * a^2 * e2 * r3
        - a^2 * r3
        + 3 * a * wbar^2 * e1
        - 3 * a * wbar^2 * e2
        - 2 * a * wbar * r3bar * e1
        + 2 * a * wbar * r3bar * e2
        + 3 * a * wbar * e1 * r3
        - 3 * a * wbar * e2 * r3
        - 3 * a * r3bar * e1 * r3
        + 3 * a * r3bar * e2 * r3
        - a * e1
        + a * e2
        + 3 * w * wbar^3 * r3
        - 3 * w * wbar^2 * r3bar * r3
        - 3 * w * wbar^2
        + 2 * w * wbar * r3bar
        + 3 * wbar^3 * r3^2
        - 3 * wbar^2 * r3bar * r3^2
        - 3 * wbar^2 * e1 * r3
        - 3 * wbar^2 * e2 * r3
        - 3 * wbar^2 * r3
        - wbar * r3bar * e1 * r3
        - wbar * r3bar * e2 * r3
        + 4 * wbar * r3bar * r3
        + wbar
        + r3bar * e1
        + r3bar * e2
        - r3bar
    )

    p_3 = (
        -3 * a^4 * w * wbar
        + 3 * a^4 * w * r3bar
        - 9 * a^4 * wbar * r3
        + 9 * a^4 * r3bar * r3
        + 2 * a^4 * e1
        + 2 * a^4 * e2
        + a^3 * w * e1
        - a^3 * w * e2
        + 3 * a^3 * wbar * e1
        - 3 * a^3 * wbar * e2
        - 3 * a^3 * r3bar * e1
        + 3 * a^3 * r3bar * e2
        + 3 * a^3 * e1 * r3
        - 3 * a^3 * e2 * r3
        + 3 * a^2 * w * wbar^3
        - 3 * a^2 * w * wbar^2 * r3bar
        + 3 * a^2 * w * wbar * r3^2
        - 3 * a^2 * w * r3bar * r3^2
        - a^2 * w * e1 * r3
        - a^2 * w * e2 * r3
        - 2 * a^2 * w * r3
        + 9 * a^2 * wbar^3 * r3
        - 9 * a^2 * wbar^2 * r3bar * r3
        + 3 * a^2 * wbar^2 * e1
        + 3 * a^2 * wbar^2 * e2
        - 6 * a^2 * wbar^2
        - 5 * a^2 * wbar * r3bar * e1
        - 5 * a^2 * wbar * r3bar * e2
        + 6 * a^2 * wbar * r3bar
        + a^2 * wbar * r3^3
        - a^2 * r3bar * r3^3
        - a^2 * e1^2
        + 2 * a^2 * e1 * e2
        - 5 * a^2 * e1 * r3^2
        - a^2 * e2^2
        - 5 * a^2 * e2 * r3^2
        + 2 * a^2 * r3^2
        - 3 * a * w * wbar^2 * e1
        + 3 * a * w * wbar^2 * e2
        + 2 * a * w * wbar * r3bar * e1
        - 2 * a * w * wbar * r3bar * e2
        - 9 * a * wbar^2 * e1 * r3
        + 9 * a * wbar^2 * e2 * r3
        + 6 * a * wbar * r3bar * e1 * r3
        - 6 * a * wbar * r3bar * e2 * r3
        - 3 * a * wbar * e1 * r3^2
        + 4 * a * wbar * e1
        + 3 * a * wbar * e2 * r3^2
        - 4 * a * wbar * e2
        + a * r3bar * e1^2
        + 3 * a * r3bar * e1 * r3^2
        - 2 * a * r3bar * e1
        - a * r3bar * e2^2
        - 3 * a * r3bar * e2 * r3^2
        + 2 * a * r3bar * e2
        + a * e1^2 * r3
        + 2 * a * e1 * r3
        - a * e2^2 * r3
        - 2 * a * e2 * r3
        - 3 * w * wbar^3 * r3^2
        + 3 * w * wbar^2 * r3bar * r3^2
        + 3 * w * wbar^2 * e1 * r3
        + 3 * w * wbar^2 * e2 * r3
        + 6 * w * wbar^2 * r3
        - 2 * w * wbar * r3bar * e1 * r3
        - 2 * w * wbar * r3bar * e2 * r3
        - 4 * w * wbar * r3bar * r3
        - 3 * w * wbar
        + w * r3bar
        - wbar^3 * r3^3
        + wbar^2 * r3bar * r3^3
        + 6 * wbar^2 * e1 * r3^2
        + 6 * wbar^2 * e2 * r3^2
        - wbar * r3bar * e1 * r3^2
        - wbar * r3bar * e2 * r3^2
        - 2 * wbar * r3bar * r3^2
        - 4 * wbar * e1 * r3
        - 4 * wbar * e2 * r3
        + wbar * r3
        - r3bar * e1^2 * r3
        - 2 * r3bar * e1 * e2 * r3
        - r3bar * e2^2 * r3
        + r3bar * r3
    )

    p_4 = (
        -3 * a^6 * wbar
        + 3 * a^6 * r3bar
        + 2 * a^5 * e1
        - 2 * a^5 * e2
        + 9 * a^4 * w * wbar * r3
        - 9 * a^4 * w * r3bar * r3
        + a^4 * w * e1
        + a^4 * w * e2
        - 3 * a^4 * w
        + 3 * a^4 * wbar^3
        - 3 * a^4 * wbar^2 * r3bar
        + 9 * a^4 * wbar * r3^2
        - 9 * a^4 * r3bar * r3^2
        - 9 * a^4 * e1 * r3
        - 9 * a^4 * e2 * r3
        + 3 * a^4 * r3
        - 3 * a^3 * w * e1 * r3
        + 3 * a^3 * w * e2 * r3
        - 6 * a^3 * wbar^2 * e1
        + 6 * a^3 * wbar^2 * e2
        + 4 * a^3 * wbar * r3bar * e1
        - 4 * a^3 * wbar * r3bar * e2
        - 9 * a^3 * wbar * e1 * r3
        + 9 * a^3 * wbar * e2 * r3
        + 9 * a^3 * r3bar * e1 * r3
        - 9 * a^3 * r3bar * e2 * r3
        - a^3 * e1^2
        - 3 * a^3 * e1 * r3^2
        + 3 * a^3 * e1
        + a^3 * e2^2
        + 3 * a^3 * e2 * r3^2
        - 3 * a^3 * e2
        - 9 * a^2 * w * wbar^3 * r3
        + 9 * a^2 * w * wbar^2 * r3bar * r3
        - 3 * a^2 * w * wbar^2 * e1
        - 3 * a^2 * w * wbar^2 * e2
        + 9 * a^2 * w * wbar^2
        + 2 * a^2 * w * wbar * r3bar * e1
        + 2 * a^2 * w * wbar * r3bar * e2
        - 6 * a^2 * w * wbar * r3bar
        - a^2 * w * wbar * r3^3
        + a^2 * w * r3bar * r3^3
        + 2 * a^2 * w * e1 * r3^2
        + 2 * a^2 * w * e2 * r3^2
        + a^2 * w * r3^2
        - 9 * a^2 * wbar^3 * r3^2
        + 9 * a^2 * wbar^2 * r3bar * r3^2
        + 9 * a^2 * wbar^2 * r3
        + 9 * a^2 * wbar * r3bar * e1 * r3
        + 9 * a^2 * wbar * r3bar * e2 * r3
        - 12 * a^2 * wbar * r3bar * r3
        + 3 * a^2 * wbar * e1^2
        - 6 * a^2 * wbar * e1 * e2
        + 4 * a^2 * wbar * e1
        + 3 * a^2 * wbar * e2^2
        + 4 * a^2 * wbar * e2
        - 3 * a^2 * wbar
        + 4 * a^2 * r3bar * e1 * e2
        - 5 * a^2 * r3bar * e1
        - 5 * a^2 * r3bar * e2
        + 3 * a^2 * r3bar
        + 3 * a^2 * e1^2 * r3
        - 6 * a^2 * e1 * e2 * r3
        + 2 * a^2 * e1 * r3^3
        + 3 * a^2 * e2^2 * r3
        + 2 * a^2 * e2 * r3^3
        - a^2 * r3^3
        + 9 * a * w * wbar^2 * e1 * r3
        - 9 * a * w * wbar^2 * e2 * r3
        - 6 * a * w * wbar * r3bar * e1 * r3
        + 6 * a * w * wbar * r3bar * e2 * r3
        - 6 * a * w * wbar * e1
        + 6 * a * w * wbar * e2
        + 2 * a * w * r3bar * e1
        - 2 * a * w * r3bar * e2
        + 9 * a * wbar^2 * e1 * r3^2
        - 9 * a * wbar^2 * e2 * r3^2
        - 6 * a * wbar * r3bar * e1 * r3^2
        + 6 * a * wbar * r3bar * e2 * r3^2
        - 6 * a * wbar * e1^2 * r3
        + a * wbar * e1 * r3^3
        - 6 * a * wbar * e1 * r3
        + 6 * a * wbar * e2^2 * r3
        - a * wbar * e2 * r3^3
        + 6 * a * wbar * e2 * r3
        - a * r3bar * e1^2 * r3
        - a * r3bar * e1 * r3^3
        + 4 * a * r3bar * e1 * r3
        + a * r3bar * e2^2 * r3
        + a * r3bar * e2 * r3^3
        - 4 * a * r3bar * e2 * r3
        - 2 * a * e1^2 * r3^2
        - a * e1 * r3^2
        + a * e1
        + 2 * a * e2^2 * r3^2
        + a * e2 * r3^2
        - a * e2
        + w * wbar^3 * r3^3
        - w * wbar^2 * r3bar * r3^3
        - 6 * w * wbar^2 * e1 * r3^2
        - 6 * w * wbar^2 * e2 * r3^2
        - 3 * w * wbar^2 * r3^2
        + 4 * w * wbar * r3bar * e1 * r3^2
        + 4 * w * wbar * r3bar * e2 * r3^2
        + 2 * w * wbar * r3bar * r3^2
        + 6 * w * wbar * e1 * r3
        + 6 * w * wbar * e2 * r3
        + 3 * w * wbar * r3
        - 2 * w * r3bar * e1 * r3
        - 2 * w * r3bar * e2 * r3
        - w * r3bar * r3
        - w
        - 3 * wbar^2 * e1 * r3^3
        - 3 * wbar^2 * e2 * r3^3
        + wbar^2 * r3^3
        + wbar * r3bar * e1 * r3^3
        + wbar * r3bar * e2 * r3^3
        + 3 * wbar * e1^2 * r3^2
        + 6 * wbar * e1 * e2 * r3^2
        + 2 * wbar * e1 * r3^2
        + 3 * wbar * e2^2 * r3^2
        + 2 * wbar * e2 * r3^2
        - 2 * wbar * r3^2
        + r3bar * e1^2 * r3^2
        + 2 * r3bar * e1 * e2 * r3^2
        - r3bar * e1 * r3^2
        + r3bar * e2^2 * r3^2
        - r3bar * e2 * r3^2
        - e1 * r3
        - e2 * r3
        + r3
    )

    p_5 = (
        3 * a^6 * w * wbar
        - 3 * a^6 * w * r3bar
        + 9 * a^6 * wbar * r3
        - 9 * a^6 * r3bar * r3
        - a^6 * e1
        - a^6 * e2
        - 2 * a^5 * w * e1
        + 2 * a^5 * w * e2
        - 3 * a^5 * wbar * e1
        + 3 * a^5 * wbar * e2
        + 3 * a^5 * r3bar * e1
        - 3 * a^5 * r3bar * e2
        - 6 * a^5 * e1 * r3
        + 6 * a^5 * e2 * r3
        - 3 * a^4 * w * wbar^3
        + 3 * a^4 * w * wbar^2 * r3bar
        - 9 * a^4 * w * wbar * r3^2
        + 9 * a^4 * w * r3bar * r3^2
        + 6 * a^4 * w * r3
        - 9 * a^4 * wbar^3 * r3
        + 9 * a^4 * wbar^2 * r3bar * r3
        - 6 * a^4 * wbar^2 * e1
        - 6 * a^4 * wbar^2 * e2
        + 6 * a^4 * wbar^2
        + 7 * a^4 * wbar * r3bar * e1
        + 7 * a^4 * wbar * r3bar * e2
        - 6 * a^4 * wbar * r3bar
        - 3 * a^4 * wbar * r3^3
        + 3 * a^4 * r3bar * r3^3
        + 2 * a^4 * e1^2
        - 4 * a^4 * e1 * e2
        + 12 * a^4 * e1 * r3^2
        + 2 * a^4 * e2^2
        + 12 * a^4 * e2 * r3^2
        - 6 * a^4 * r3^2
        + 6 * a^3 * w * wbar^2 * e1
        - 6 * a^3 * w * wbar^2 * e2
        - 4 * a^3 * w * wbar * r3bar * e1
        + 4 * a^3 * w * wbar * r3bar * e2
        + 3 * a^3 * w * e1 * r3^2
        - 3 * a^3 * w * e2 * r3^2
        + 18 * a^3 * wbar^2 * e1 * r3
        - 18 * a^3 * wbar^2 * e2 * r3
        - 12 * a^3 * wbar * r3bar * e1 * r3
        + 12 * a^3 * wbar * r3bar * e2 * r3
        + 6 * a^3 * wbar * e1^2
        + 9 * a^3 * wbar * e1 * r3^2
        - 8 * a^3 * wbar * e1
        - 6 * a^3 * wbar * e2^2
        - 9 * a^3 * wbar * e2 * r3^2
        + 8 * a^3 * wbar * e2
        - 4 * a^3 * r3bar * e1^2
        - 9 * a^3 * r3bar * e1 * r3^2
        + 4 * a^3 * r3bar * e1
        + 4 * a^3 * r3bar * e2^2
        + 9 * a^3 * r3bar * e2 * r3^2
        - 4 * a^3 * r3bar * e2
        + a^3 * e1 * r3^3
        - 6 * a^3 * e1 * r3
        - a^3 * e2 * r3^3
        + 6 * a^3 * e2 * r3
        + 9 * a^2 * w * wbar^3 * r3^2
        - 9 * a^2 * w * wbar^2 * r3bar * r3^2
        - 18 * a^2 * w * wbar^2 * r3
        + 12 * a^2 * w * wbar * r3bar * r3
        - 3 * a^2 * w * wbar * e1^2
        + 6 * a^2 * w * wbar * e1 * e2
        - 6 * a^2 * w * wbar * e1
        - 3 * a^2 * w * wbar * e2^2
        - 6 * a^2 * w * wbar * e2
        + 9 * a^2 * w * wbar
        + a^2 * w * r3bar * e1^2
        - 2 * a^2 * w * r3bar * e1 * e2
        + 2 * a^2 * w * r3bar * e1
        + a^2 * w * r3bar * e2^2
        + 2 * a^2 * w * r3bar * e2
        - 3 * a^2 * w * r3bar
        - a^2 * w * e1 * r3^3
        - a^2 * w * e2 * r3^3
        + 3 * a^2 * wbar^3 * r3^3
        - 3 * a^2 * wbar^2 * r3bar * r3^3
        - 9 * a^2 * wbar^2 * e1 * r3^2
        - 9 * a^2 * wbar^2 * e2 * r3^2
        - 3 * a^2 * wbar * r3bar * e1 * r3^2
        - 3 * a^2 * wbar * r3bar * e2 * r3^2
        + 6 * a^2 * wbar * r3bar * r3^2
        - 15 * a^2 * wbar * e1^2 * r3
        + 6 * a^2 * wbar * e1 * e2 * r3
        + 6 * a^2 * wbar * e1 * r3
        - 15 * a^2 * wbar * e2^2 * r3
        + 6 * a^2 * wbar * e2 * r3
        - 3 * a^2 * wbar * r3
        + 5 * a^2 * r3bar * e1^2 * r3
        - 2 * a^2 * r3bar * e1 * e2 * r3
        + 4 * a^2 * r3bar * e1 * r3
        + 5 * a^2 * r3bar * e2^2 * r3
        + 4 * a^2 * r3bar * e2 * r3
        - 3 * a^2 * r3bar * r3
        - 3 * a^2 * e1^2 * r3^2
        + 2 * a^2 * e1^2
        + 6 * a^2 * e1 * e2 * r3^2
        - 4 * a^2 * e1 * e2
        + a^2 * e1
        - 3 * a^2 * e2^2 * r3^2
        + 2 * a^2 * e2^2
        + a^2 * e2
        - 9 * a * w * wbar^2 * e1 * r3^2
        + 9 * a * w * wbar^2 * e2 * r3^2
        + 6 * a * w * wbar * r3bar * e1 * r3^2
        - 6 * a * w * wbar * r3bar * e2 * r3^2
        + 6 * a * w * wbar * e1^2 * r3
        + 12 * a * w * wbar * e1 * r3
        - 6 * a * w * wbar * e2^2 * r3
        - 12 * a * w * wbar * e2 * r3
        - 2 * a * w * r3bar * e1^2 * r3
        - 4 * a * w * r3bar * e1 * r3
        + 2 * a * w * r3bar * e2^2 * r3
        + 4 * a * w * r3bar * e2 * r3
        - 3 * a * w * e1
        + 3 * a * w * e2
        - 3 * a * wbar^2 * e1 * r3^3
        + 3 * a * wbar^2 * e2 * r3^3
        + 2 * a * wbar * r3bar * e1 * r3^3
        - 2 * a * wbar * r3bar * e2 * r3^3
        + 12 * a * wbar * e1^2 * r3^2
        - 12 * a * wbar * e2^2 * r3^2
        - a * r3bar * e1^2 * r3^2
        - 2 * a * r3bar * e1 * r3^2
        + a * r3bar * e2^2 * r3^2
        + 2 * a * r3bar * e2 * r3^2
        + a * e1^2 * r3^3
        - 4 * a * e1^2 * r3
        + a * e1 * r3
        - a * e2^2 * r3^3
        + 4 * a * e2^2 * r3
        - a * e2 * r3
        + 3 * w * wbar^2 * e1 * r3^3
        + 3 * w * wbar^2 * e2 * r3^3
        - 2 * w * wbar * r3bar * e1 * r3^3
        - 2 * w * wbar * r3bar * e2 * r3^3
        - 3 * w * wbar * e1^2 * r3^2
        - 6 * w * wbar * e1 * e2 * r3^2
        - 6 * w * wbar * e1 * r3^2
        - 3 * w * wbar * e2^2 * r3^2
        - 6 * w * wbar * e2 * r3^2
        + w * r3bar * e1^2 * r3^2
        + 2 * w * r3bar * e1 * e2 * r3^2
        + 2 * w * r3bar * e1 * r3^2
        + w * r3bar * e2^2 * r3^2
        + 2 * w * r3bar * e2 * r3^2
        + 3 * w * e1 * r3
        + 3 * w * e2 * r3
        - 3 * wbar * e1^2 * r3^3
        - 6 * wbar * e1 * e2 * r3^3
        + 2 * wbar * e1 * r3^3
        - 3 * wbar * e2^2 * r3^3
        + 2 * wbar * e2 * r3^3
        + 2 * e1^2 * r3^2
        + 4 * e1 * e2 * r3^2
        - 2 * e1 * r3^2
        + 2 * e2^2 * r3^2
        - 2 * e2 * r3^2
    )

    p_6 = (
        a^8 * wbar
        - a^8 * r3bar
        - a^7 * e1
        + a^7 * e2
        - 9 * a^6 * w * wbar * r3
        + 9 * a^6 * w * r3bar * r3
        - 2 * a^6 * w * e1
        - 2 * a^6 * w * e2
        + 3 * a^6 * w
        - a^6 * wbar^3
        + a^6 * wbar^2 * r3bar
        - 9 * a^6 * wbar * r3^2
        + 9 * a^6 * r3bar * r3^2
        + 6 * a^6 * e1 * r3
        + 6 * a^6 * e2 * r3
        - 3 * a^6 * r3
        + 6 * a^5 * w * e1 * r3
        - 6 * a^5 * w * e2 * r3
        + 3 * a^5 * wbar^2 * e1
        - 3 * a^5 * wbar^2 * e2
        - 2 * a^5 * wbar * r3bar * e1
        + 2 * a^5 * wbar * r3bar * e2
        + 9 * a^5 * wbar * e1 * r3
        - 9 * a^5 * wbar * e2 * r3
        - 9 * a^5 * r3bar * e1 * r3
        + 9 * a^5 * r3bar * e2 * r3
        + 2 * a^5 * e1^2
        + 6 * a^5 * e1 * r3^2
        - 3 * a^5 * e1
        - 2 * a^5 * e2^2
        - 6 * a^5 * e2 * r3^2
        + 3 * a^5 * e2
        + 9 * a^4 * w * wbar^3 * r3
        - 9 * a^4 * w * wbar^2 * r3bar * r3
        + 6 * a^4 * w * wbar^2 * e1
        + 6 * a^4 * w * wbar^2 * e2
        - 9 * a^4 * w * wbar^2
        - 4 * a^4 * w * wbar * r3bar * e1
        - 4 * a^4 * w * wbar * r3bar * e2
        + 6 * a^4 * w * wbar * r3bar
        + 3 * a^4 * w * wbar * r3^3
        - 3 * a^4 * w * r3bar * r3^3
        - 3 * a^4 * w * e1 * r3^2
        - 3 * a^4 * w * e2 * r3^2
        - 3 * a^4 * w * r3^2
        + 9 * a^4 * wbar^3 * r3^2
        - 9 * a^4 * wbar^2 * r3bar * r3^2
        + 9 * a^4 * wbar^2 * e1 * r3
        + 9 * a^4 * wbar^2 * e2 * r3
        - 9 * a^4 * wbar^2 * r3
        - 15 * a^4 * wbar * r3bar * e1 * r3
        - 15 * a^4 * wbar * r3bar * e2 * r3
        + 12 * a^4 * wbar * r3bar * r3
        + 12 * a^4 * wbar * e1 * e2
        - 8 * a^4 * wbar * e1
        - 8 * a^4 * wbar * e2
        + 3 * a^4 * wbar
        - 2 * a^4 * r3bar * e1^2
        - 8 * a^4 * r3bar * e1 * e2
        + 7 * a^4 * r3bar * e1
        - 2 * a^4 * r3bar * e2^2
        + 7 * a^4 * r3bar * e2
        - 3 * a^4 * r3bar
        - 6 * a^4 * e1^2 * r3
        + 12 * a^4 * e1 * e2 * r3
        - 5 * a^4 * e1 * r3^3
        - 6 * a^4 * e2^2 * r3
        - 5 * a^4 * e2 * r3^3
        + 3 * a^4 * r3^3
        - 18 * a^3 * w * wbar^2 * e1 * r3
        + 18 * a^3 * w * wbar^2 * e2 * r3
        + 12 * a^3 * w * wbar * r3bar * e1 * r3
        - 12 * a^3 * w * wbar * r3bar * e2 * r3
        - 6 * a^3 * w * wbar * e1^2
        + 12 * a^3 * w * wbar * e1
        + 6 * a^3 * w * wbar * e2^2
        - 12 * a^3 * w * wbar * e2
        + 2 * a^3 * w * r3bar * e1^2
        - 4 * a^3 * w * r3bar * e1
        - 2 * a^3 * w * r3bar * e2^2
        + 4 * a^3 * w * r3bar * e2
        - a^3 * w * e1 * r3^3
        + a^3 * w * e2 * r3^3
        - 18 * a^3 * wbar^2 * e1 * r3^2
        + 18 * a^3 * wbar^2 * e2 * r3^2
        + 12 * a^3 * wbar * r3bar * e1 * r3^2
        - 12 * a^3 * wbar * r3bar * e2 * r3^2
        - 6 * a^3 * wbar * e1^2 * r3
        - 3 * a^3 * wbar * e1 * r3^3
        + 12 * a^3 * wbar * e1 * r3
        + 6 * a^3 * wbar * e2^2 * r3
        + 3 * a^3 * wbar * e2 * r3^3
        - 12 * a^3 * wbar * e2 * r3
        + 8 * a^3 * r3bar * e1^2 * r3
        + 3 * a^3 * r3bar * e1 * r3^3
        - 8 * a^3 * r3bar * e1 * r3
        - 8 * a^3 * r3bar * e2^2 * r3
        - 3 * a^3 * r3bar * e2 * r3^3
        + 8 * a^3 * r3bar * e2 * r3
        + a^3 * e1^3
        - 3 * a^3 * e1^2 * e2
        + 3 * a^3 * e1^2 * r3^2
        + 4 * a^3 * e1^2
        + 3 * a^3 * e1 * e2^2
        + 3 * a^3 * e1 * r3^2
        - 2 * a^3 * e1
        - a^3 * e2^3
        - 3 * a^3 * e2^2 * r3^2
        - 4 * a^3 * e2^2
        - 3 * a^3 * e2 * r3^2
        + 2 * a^3 * e2
        - 3 * a^2 * w * wbar^3 * r3^3
        + 3 * a^2 * w * wbar^2 * r3bar * r3^3
        + 9 * a^2 * w * wbar^2 * e1 * r3^2
        + 9 * a^2 * w * wbar^2 * e2 * r3^2
        + 9 * a^2 * w * wbar^2 * r3^2
        - 6 * a^2 * w * wbar * r3bar * e1 * r3^2
        - 6 * a^2 * w * wbar * r3bar * e2 * r3^2
        - 6 * a^2 * w * wbar * r3bar * r3^2
        + 15 * a^2 * w * wbar * e1^2 * r3
        - 6 * a^2 * w * wbar * e1 * e2 * r3
        - 6 * a^2 * w * wbar * e1 * r3
        + 15 * a^2 * w * wbar * e2^2 * r3
        - 6 * a^2 * w * wbar * e2 * r3
        - 9 * a^2 * w * wbar * r3
        - 5 * a^2 * w * r3bar * e1^2 * r3
        + 2 * a^2 * w * r3bar * e1 * e2 * r3
        + 2 * a^2 * w * r3bar * e1 * r3
        - 5 * a^2 * w * r3bar * e2^2 * r3
        + 2 * a^2 * w * r3bar * e2 * r3
        + 3 * a^2 * w * r3bar * r3
        - 3 * a^2 * w * e1^2
        + 6 * a^2 * w * e1 * e2
        - 3 * a^2 * w * e1
        - 3 * a^2 * w * e2^2
        - 3 * a^2 * w * e2
        + 3 * a^2 * w
        + 6 * a^2 * wbar^2 * e1 * r3^3
        + 6 * a^2 * wbar^2 * e2 * r3^3
        - 3 * a^2 * wbar^2 * r3^3
        - a^2 * wbar * r3bar * e1 * r3^3
        - a^2 * wbar * r3bar * e2 * r3^3
        + 12 * a^2 * wbar * e1^2 * r3^2
        - 12 * a^2 * wbar * e1 * e2 * r3^2
        - 6 * a^2 * wbar * e1 * r3^2
        + 12 * a^2 * wbar * e2^2 * r3^2
        - 6 * a^2 * wbar * e2 * r3^2
        + 6 * a^2 * wbar * r3^2
        - 7 * a^2 * r3bar * e1^2 * r3^2
        - 2 * a^2 * r3bar * e1 * e2 * r3^2
        + a^2 * r3bar * e1 * r3^2
        - 7 * a^2 * r3bar * e2^2 * r3^2
        + a^2 * r3bar * e2 * r3^2
        - 3 * a^2 * e1^3 * r3
        + 3 * a^2 * e1^2 * e2 * r3
        + a^2 * e1^2 * r3^3
        - 7 * a^2 * e1^2 * r3
        + 3 * a^2 * e1 * e2^2 * r3
        - 2 * a^2 * e1 * e2 * r3^3
        - 2 * a^2 * e1 * e2 * r3
        + 4 * a^2 * e1 * r3
        - 3 * a^2 * e2^3 * r3
        + a^2 * e2^2 * r3^3
        - 7 * a^2 * e2^2 * r3
        + 4 * a^2 * e2 * r3
        - 3 * a^2 * r3
        + 3 * a * w * wbar^2 * e1 * r3^3
        - 3 * a * w * wbar^2 * e2 * r3^3
        - 2 * a * w * wbar * r3bar * e1 * r3^3
        + 2 * a * w * wbar * r3bar * e2 * r3^3
        - 12 * a * w * wbar * e1^2 * r3^2
        - 6 * a * w * wbar * e1 * r3^2
        + 12 * a * w * wbar * e2^2 * r3^2
        + 6 * a * w * wbar * e2 * r3^2
        + 4 * a * w * r3bar * e1^2 * r3^2
        + 2 * a * w * r3bar * e1 * r3^2
        - 4 * a * w * r3bar * e2^2 * r3^2
        - 2 * a * w * r3bar * e2 * r3^2
        + 6 * a * w * e1^2 * r3
        + 3 * a * w * e1 * r3
        - 6 * a * w * e2^2 * r3
        - 3 * a * w * e2 * r3
        - 6 * a * wbar * e1^2 * r3^3
        + 2 * a * wbar * e1 * r3^3
        + 6 * a * wbar * e2^2 * r3^3
        - 2 * a * wbar * e2 * r3^3
        + a * r3bar * e1^2 * r3^3
        - a * r3bar * e2^2 * r3^3
        + 3 * a * e1^3 * r3^2
        + 3 * a * e1^2 * e2 * r3^2
        + 2 * a * e1^2 * r3^2
        - 3 * a * e1 * e2^2 * r3^2
        - 2 * a * e1 * r3^2
        - 3 * a * e2^3 * r3^2
        - 2 * a * e2^2 * r3^2
        + 2 * a * e2 * r3^2
        + 3 * w * wbar * e1^2 * r3^3
        + 6 * w * wbar * e1 * e2 * r3^3
        + 3 * w * wbar * e2^2 * r3^3
        - w * r3bar * e1^2 * r3^3
        - 2 * w * r3bar * e1 * e2 * r3^3
        - w * r3bar * e2^2 * r3^3
        - 3 * w * e1^2 * r3^2
        - 6 * w * e1 * e2 * r3^2
        - 3 * w * e2^2 * r3^2
        - e1^3 * r3^3
        - 3 * e1^2 * e2 * r3^3
        + e1^2 * r3^3
        - 3 * e1 * e2^2 * r3^3
        + 2 * e1 * e2 * r3^3
        - e2^3 * r3^3
        + e2^2 * r3^3
    )

    p_7 = (
        -(a^8) * w * wbar
        + a^8 * w * r3bar
        - 3 * a^8 * wbar * r3
        + 3 * a^8 * r3bar * r3
        + a^7 * w * e1
        - a^7 * w * e2
        + a^7 * wbar * e1
        - a^7 * wbar * e2
        - a^7 * r3bar * e1
        + a^7 * r3bar * e2
        + 3 * a^7 * e1 * r3
        - 3 * a^7 * e2 * r3
        + a^6 * w * wbar^3
        - a^6 * w * wbar^2 * r3bar
        + 9 * a^6 * w * wbar * r3^2
        - 9 * a^6 * w * r3bar * r3^2
        + 3 * a^6 * w * e1 * r3
        + 3 * a^6 * w * e2 * r3
        - 6 * a^6 * w * r3
        + 3 * a^6 * wbar^3 * r3
        - 3 * a^6 * wbar^2 * r3bar * r3
        + 3 * a^6 * wbar^2 * e1
        + 3 * a^6 * wbar^2 * e2
        - 2 * a^6 * wbar^2
        - 3 * a^6 * wbar * r3bar * e1
        - 3 * a^6 * wbar * r3bar * e2
        + 2 * a^6 * wbar * r3bar
        + 3 * a^6 * wbar * r3^3
        - 3 * a^6 * r3bar * r3^3
        - a^6 * e1^2
        + 2 * a^6 * e1 * e2
        - 9 * a^6 * e1 * r3^2
        - a^6 * e2^2
        - 9 * a^6 * e2 * r3^2
        + 6 * a^6 * r3^2
        - 3 * a^5 * w * wbar^2 * e1
        + 3 * a^5 * w * wbar^2 * e2
        + 2 * a^5 * w * wbar * r3bar * e1
        - 2 * a^5 * w * wbar * r3bar * e2
        - 6 * a^5 * w * e1 * r3^2
        + 6 * a^5 * w * e2 * r3^2
        - 9 * a^5 * wbar^2 * e1 * r3
        + 9 * a^5 * wbar^2 * e2 * r3
        + 6 * a^5 * wbar * r3bar * e1 * r3
        - 6 * a^5 * wbar * r3bar * e2 * r3
        - 6 * a^5 * wbar * e1^2
        - 9 * a^5 * wbar * e1 * r3^2
        + 4 * a^5 * wbar * e1
        + 6 * a^5 * wbar * e2^2
        + 9 * a^5 * wbar * e2 * r3^2
        - 4 * a^5 * wbar * e2
        + 3 * a^5 * r3bar * e1^2
        + 9 * a^5 * r3bar * e1 * r3^2
        - 2 * a^5 * r3bar * e1
        - 3 * a^5 * r3bar * e2^2
        - 9 * a^5 * r3bar * e2 * r3^2
        + 2 * a^5 * r3bar * e2
        - 3 * a^5 * e1^2 * r3
        - 2 * a^5 * e1 * r3^3
        + 6 * a^5 * e1 * r3
        + 3 * a^5 * e2^2 * r3
        + 2 * a^5 * e2 * r3^3
        - 6 * a^5 * e2 * r3
        - 9 * a^4 * w * wbar^3 * r3^2
        + 9 * a^4 * w * wbar^2 * r3bar * r3^2
        - 9 * a^4 * w * wbar^2 * e1 * r3
        - 9 * a^4 * w * wbar^2 * e2 * r3
        + 18 * a^4 * w * wbar^2 * r3
        + 6 * a^4 * w * wbar * r3bar * e1 * r3
        + 6 * a^4 * w * wbar * r3bar * e2 * r3
        - 12 * a^4 * w * wbar * r3bar * r3
        - 12 * a^4 * w * wbar * e1 * e2
        + 12 * a^4 * w * wbar * e1
        + 12 * a^4 * w * wbar * e2
        - 9 * a^4 * w * wbar
        + 4 * a^4 * w * r3bar * e1 * e2
        - 4 * a^4 * w * r3bar * e1
        - 4 * a^4 * w * r3bar * e2
        + 3 * a^4 * w * r3bar
        + 2 * a^4 * w * e1 * r3^3
        + 2 * a^4 * w * e2 * r3^3
        - 3 * a^4 * wbar^3 * r3^3
        + 3 * a^4 * wbar^2 * r3bar * r3^3
        + 9 * a^4 * wbar * r3bar * e1 * r3^2
        + 9 * a^4 * wbar * r3bar * e2 * r3^2
        - 6 * a^4 * wbar * r3bar * r3^2
        + 12 * a^4 * wbar * e1^2 * r3
        - 12 * a^4 * wbar * e1 * e2 * r3
        + 12 * a^4 * wbar * e2^2 * r3
        + 3 * a^4 * wbar * r3
        - a^4 * r3bar * e1^2 * r3
        + 10 * a^4 * r3bar * e1 * e2 * r3
        - 8 * a^4 * r3bar * e1 * r3
        - a^4 * r3bar * e2^2 * r3
        - 8 * a^4 * r3bar * e2 * r3
        + 3 * a^4 * r3bar * r3
        + 3 * a^4 * e1^3
        - 3 * a^4 * e1^2 * e2
        + 6 * a^4 * e1^2 * r3^2
        - 3 * a^4 * e1 * e2^2
        - 12 * a^4 * e1 * e2 * r3^2
        + 8 * a^4 * e1 * e2
        - 2 * a^4 * e1
        + 3 * a^4 * e2^3
        + 6 * a^4 * e2^2 * r3^2
        - 2 * a^4 * e2
        + 18 * a^3 * w * wbar^2 * e1 * r3^2
        - 18 * a^3 * w * wbar^2 * e2 * r3^2
        - 12 * a^3 * w * wbar * r3bar * e1 * r3^2
        + 12 * a^3 * w * wbar * r3bar * e2 * r3^2
        + 6 * a^3 * w * wbar * e1^2 * r3
        - 24 * a^3 * w * wbar * e1 * r3
        - 6 * a^3 * w * wbar * e2^2 * r3
        + 24 * a^3 * w * wbar * e2 * r3
        - 2 * a^3 * w * r3bar * e1^2 * r3
        + 8 * a^3 * w * r3bar * e1 * r3
        + 2 * a^3 * w * r3bar * e2^2 * r3
        - 8 * a^3 * w * r3bar * e2 * r3
        - a^3 * w * e1^3
        + 3 * a^3 * w * e1^2 * e2
        - 6 * a^3 * w * e1^2
        - 3 * a^3 * w * e1 * e2^2
        + 6 * a^3 * w * e1
        + a^3 * w * e2^3
        + 6 * a^3 * w * e2^2
        - 6 * a^3 * w * e2
        + 6 * a^3 * wbar^2 * e1 * r3^3
        - 6 * a^3 * wbar^2 * e2 * r3^3
        - 4 * a^3 * wbar * r3bar * e1 * r3^3
        + 4 * a^3 * wbar * r3bar * e2 * r3^3
        - 6 * a^3 * wbar * e1^2 * r3^2
        + 6 * a^3 * wbar * e2^2 * r3^2
        - 4 * a^3 * r3bar * e1^2 * r3^2
        + 4 * a^3 * r3bar * e1 * r3^2
        + 4 * a^3 * r3bar * e2^2 * r3^2
        - 4 * a^3 * r3bar * e2 * r3^2
        - 9 * a^3 * e1^3 * r3
        + 3 * a^3 * e1^2 * e2 * r3
        - 2 * a^3 * e1^2 * r3^3
        + 2 * a^3 * e1^2 * r3
        - 3 * a^3 * e1 * e2^2 * r3
        - 2 * a^3 * e1 * r3
        + 9 * a^3 * e2^3 * r3
        + 2 * a^3 * e2^2 * r3^3
        - 2 * a^3 * e2^2 * r3
        + 2 * a^3 * e2 * r3
        - 6 * a^2 * w * wbar^2 * e1 * r3^3
        - 6 * a^2 * w * wbar^2 * e2 * r3^3
        + 4 * a^2 * w * wbar * r3bar * e1 * r3^3
        + 4 * a^2 * w * wbar * r3bar * e2 * r3^3
        - 12 * a^2 * w * wbar * e1^2 * r3^2
        + 12 * a^2 * w * wbar * e1 * e2 * r3^2
        + 12 * a^2 * w * wbar * e1 * r3^2
        - 12 * a^2 * w * wbar * e2^2 * r3^2
        + 12 * a^2 * w * wbar * e2 * r3^2
        + 4 * a^2 * w * r3bar * e1^2 * r3^2
        - 4 * a^2 * w * r3bar * e1 * e2 * r3^2
        - 4 * a^2 * w * r3bar * e1 * r3^2
        + 4 * a^2 * w * r3bar * e2^2 * r3^2
        - 4 * a^2 * w * r3bar * e2 * r3^2
        + 3 * a^2 * w * e1^3 * r3
        - 3 * a^2 * w * e1^2 * e2 * r3
        + 12 * a^2 * w * e1^2 * r3
        - 3 * a^2 * w * e1 * e2^2 * r3
        - 6 * a^2 * w * e1 * r3
        + 3 * a^2 * w * e2^3 * r3
        + 12 * a^2 * w * e2^2 * r3
        - 6 * a^2 * w * e2 * r3
        + 12 * a^2 * wbar * e1 * e2 * r3^3
        - 4 * a^2 * wbar * e1 * r3^3
        - 4 * a^2 * wbar * e2 * r3^3
        + 2 * a^2 * r3bar * e1^2 * r3^3
        + 2 * a^2 * r3bar * e2^2 * r3^3
        + 9 * a^2 * e1^3 * r3^2
        + 3 * a^2 * e1^2 * e2 * r3^2
        - 4 * a^2 * e1^2 * r3^2
        + 3 * a^2 * e1 * e2^2 * r3^2
        - 8 * a^2 * e1 * e2 * r3^2
        + 4 * a^2 * e1 * r3^2
        + 9 * a^2 * e2^3 * r3^2
        - 4 * a^2 * e2^2 * r3^2
        + 4 * a^2 * e2 * r3^2
        + 6 * a * w * wbar * e1^2 * r3^3
        - 6 * a * w * wbar * e2^2 * r3^3
        - 2 * a * w * r3bar * e1^2 * r3^3
        + 2 * a * w * r3bar * e2^2 * r3^3
        - 3 * a * w * e1^3 * r3^2
        - 3 * a * w * e1^2 * e2 * r3^2
        - 6 * a * w * e1^2 * r3^2
        + 3 * a * w * e1 * e2^2 * r3^2
        + 3 * a * w * e2^3 * r3^2
        + 6 * a * w * e2^2 * r3^2
        - 3 * a * e1^3 * r3^3
        - 3 * a * e1^2 * e2 * r3^3
        + 2 * a * e1^2 * r3^3
        + 3 * a * e1 * e2^2 * r3^3
        + 3 * a * e2^3 * r3^3
        - 2 * a * e2^2 * r3^3
        + w * e1^3 * r3^3
        + 3 * w * e1^2 * e2 * r3^3
        + 3 * w * e1 * e2^2 * r3^3
        + w * e2^3 * r3^3
    )

    p_8 = (
        3 * a^8 * w * wbar * r3
        - 3 * a^8 * w * r3bar * r3
        + a^8 * w * e1
        + a^8 * w * e2
        - a^8 * w
        + 3 * a^8 * wbar * r3^2
        - 3 * a^8 * r3bar * r3^2
        - a^8 * e1 * r3
        - a^8 * e2 * r3
        + a^8 * r3
        - 3 * a^7 * w * e1 * r3
        + 3 * a^7 * w * e2 * r3
        - 3 * a^7 * wbar * e1 * r3
        + 3 * a^7 * wbar * e2 * r3
        + 3 * a^7 * r3bar * e1 * r3
        - 3 * a^7 * r3bar * e2 * r3
        - a^7 * e1^2
        - 3 * a^7 * e1 * r3^2
        + a^7 * e1
        + a^7 * e2^2
        + 3 * a^7 * e2 * r3^2
        - a^7 * e2
        - 3 * a^6 * w * wbar^3 * r3
        + 3 * a^6 * w * wbar^2 * r3bar * r3
        - 3 * a^6 * w * wbar^2 * e1
        - 3 * a^6 * w * wbar^2 * e2
        + 3 * a^6 * w * wbar^2
        + 2 * a^6 * w * wbar * r3bar * e1
        + 2 * a^6 * w * wbar * r3bar * e2
        - 2 * a^6 * w * wbar * r3bar
        - 3 * a^6 * w * wbar * r3^3
        + 3 * a^6 * w * r3bar * r3^3
        + 3 * a^6 * w * r3^2
        - 3 * a^6 * wbar^3 * r3^2
        + 3 * a^6 * wbar^2 * r3bar * r3^2
        - 6 * a^6 * wbar^2 * e1 * r3
        - 6 * a^6 * wbar^2 * e2 * r3
        + 3 * a^6 * wbar^2 * r3
        + 7 * a^6 * wbar * r3bar * e1 * r3
        + 7 * a^6 * wbar * r3bar * e2 * r3
        - 4 * a^6 * wbar * r3bar * r3
        - 3 * a^6 * wbar * e1^2
        - 6 * a^6 * wbar * e1 * e2
        + 4 * a^6 * wbar * e1
        - 3 * a^6 * wbar * e2^2
        + 4 * a^6 * wbar * e2
        - a^6 * wbar
        + 2 * a^6 * r3bar * e1^2
        + 4 * a^6 * r3bar * e1 * e2
        - 3 * a^6 * r3bar * e1
        + 2 * a^6 * r3bar * e2^2
        - 3 * a^6 * r3bar * e2
        + a^6 * r3bar
        + 3 * a^6 * e1^2 * r3
        - 6 * a^6 * e1 * e2 * r3
        + 4 * a^6 * e1 * r3^3
        + 3 * a^6 * e2^2 * r3
        + 4 * a^6 * e2 * r3^3
        - 3 * a^6 * r3^3
        + 9 * a^5 * w * wbar^2 * e1 * r3
        - 9 * a^5 * w * wbar^2 * e2 * r3
        - 6 * a^5 * w * wbar * r3bar * e1 * r3
        + 6 * a^5 * w * wbar * r3bar * e2 * r3
        + 6 * a^5 * w * wbar * e1^2
        - 6 * a^5 * w * wbar * e1
        - 6 * a^5 * w * wbar * e2^2
        + 6 * a^5 * w * wbar * e2
        - 2 * a^5 * w * r3bar * e1^2
        + 2 * a^5 * w * r3bar * e1
        + 2 * a^5 * w * r3bar * e2^2
        - 2 * a^5 * w * r3bar * e2
        + 2 * a^5 * w * e1 * r3^3
        - 2 * a^5 * w * e2 * r3^3
        + 9 * a^5 * wbar^2 * e1 * r3^2
        - 9 * a^5 * wbar^2 * e2 * r3^2
        - 6 * a^5 * wbar * r3bar * e1 * r3^2
        + 6 * a^5 * wbar * r3bar * e2 * r3^2
        + 12 * a^5 * wbar * e1^2 * r3
        + 3 * a^5 * wbar * e1 * r3^3
        - 6 * a^5 * wbar * e1 * r3
        - 12 * a^5 * wbar * e2^2 * r3
        - 3 * a^5 * wbar * e2 * r3^3
        + 6 * a^5 * wbar * e2 * r3
        - 7 * a^5 * r3bar * e1^2 * r3
        - 3 * a^5 * r3bar * e1 * r3^3
        + 4 * a^5 * r3bar * e1 * r3
        + 7 * a^5 * r3bar * e2^2 * r3
        + 3 * a^5 * r3bar * e2 * r3^3
        - 4 * a^5 * r3bar * e2 * r3
        + 3 * a^5 * e1^3
        + 3 * a^5 * e1^2 * e2
        - 4 * a^5 * e1^2
        - 3 * a^5 * e1 * e2^2
        - 3 * a^5 * e1 * r3^2
        + a^5 * e1
        - 3 * a^5 * e2^3
        + 4 * a^5 * e2^2
        + 3 * a^5 * e2 * r3^2
        - a^5 * e2
        + 3 * a^4 * w * wbar^3 * r3^3
        - 3 * a^4 * w * wbar^2 * r3bar * r3^3
        - 9 * a^4 * w * wbar^2 * r3^2
        + 6 * a^4 * w * wbar * r3bar * r3^2
        - 12 * a^4 * w * wbar * e1^2 * r3
        + 12 * a^4 * w * wbar * e1 * e2 * r3
        - 6 * a^4 * w * wbar * e1 * r3
        - 12 * a^4 * w * wbar * e2^2 * r3
        - 6 * a^4 * w * wbar * e2 * r3
        + 9 * a^4 * w * wbar * r3
        + 4 * a^4 * w * r3bar * e1^2 * r3
        - 4 * a^4 * w * r3bar * e1 * e2 * r3
        + 2 * a^4 * w * r3bar * e1 * r3
        + 4 * a^4 * w * r3bar * e2^2 * r3
        + 2 * a^4 * w * r3bar * e2 * r3
        - 3 * a^4 * w * r3bar * r3
        - 3 * a^4 * w * e1^3
        + 3 * a^4 * w * e1^2 * e2
        + 3 * a^4 * w * e1 * e2^2
        - 12 * a^4 * w * e1 * e2
        + 6 * a^4 * w * e1
        - 3 * a^4 * w * e2^3
        + 6 * a^4 * w * e2
        - 3 * a^4 * w
        - 3 * a^4 * wbar^2 * e1 * r3^3
        - 3 * a^4 * wbar^2 * e2 * r3^3
        + 3 * a^4 * wbar^2 * r3^3
        - a^4 * wbar * r3bar * e1 * r3^3
        - a^4 * wbar * r3bar * e2 * r3^3
        - 15 * a^4 * wbar * e1^2 * r3^2
        + 6 * a^4 * wbar * e1 * e2 * r3^2
        + 6 * a^4 * wbar * e1 * r3^2
        - 15 * a^4 * wbar * e2^2 * r3^2
        + 6 * a^4 * wbar * e2 * r3^2
        - 6 * a^4 * wbar * r3^2
        + 5 * a^4 * r3bar * e1^2 * r3^2
        - 2 * a^4 * r3bar * e1 * e2 * r3^2
        + a^4 * r3bar * e1 * r3^2
        + 5 * a^4 * r3bar * e2^2 * r3^2
        + a^4 * r3bar * e2 * r3^2
        - 9 * a^4 * e1^3 * r3
        - 3 * a^4 * e1^2 * e2 * r3
        - 2 * a^4 * e1^2 * r3^3
        + 8 * a^4 * e1^2 * r3
        - 3 * a^4 * e1 * e2^2 * r3
        + 4 * a^4 * e1 * e2 * r3^3
        + 4 * a^4 * e1 * e2 * r3
        - 5 * a^4 * e1 * r3
        - 9 * a^4 * e2^3 * r3
        - 2 * a^4 * e2^2 * r3^3
        + 8 * a^4 * e2^2 * r3
        - 5 * a^4 * e2 * r3
        + 3 * a^4 * r3
        - 6 * a^3 * w * wbar^2 * e1 * r3^3
        + 6 * a^3 * w * wbar^2 * e2 * r3^3
        + 4 * a^3 * w * wbar * r3bar * e1 * r3^3
        - 4 * a^3 * w * wbar * r3bar * e2 * r3^3
        + 6 * a^3 * w * wbar * e1^2 * r3^2
        + 12 * a^3 * w * wbar * e1 * r3^2
        - 6 * a^3 * w * wbar * e2^2 * r3^2
        - 12 * a^3 * w * wbar * e2 * r3^2
        - 2 * a^3 * w * r3bar * e1^2 * r3^2
        - 4 * a^3 * w * r3bar * e1 * r3^2
        + 2 * a^3 * w * r3bar * e2^2 * r3^2
        + 4 * a^3 * w * r3bar * e2 * r3^2
        + 9 * a^3 * w * e1^3 * r3
        - 3 * a^3 * w * e1^2 * e2 * r3
        + 3 * a^3 * w * e1 * e2^2 * r3
        - 6 * a^3 * w * e1 * r3
        - 9 * a^3 * w * e2^3 * r3
        + 6 * a^3 * w * e2 * r3
        + 6 * a^3 * wbar * e1^2 * r3^3
        - 4 * a^3 * wbar * e1 * r3^3
        - 6 * a^3 * wbar * e2^2 * r3^3
        + 4 * a^3 * wbar * e2 * r3^3
        + 9 * a^3 * e1^3 * r3^2
        - 3 * a^3 * e1^2 * e2 * r3^2
        - 4 * a^3 * e1^2 * r3^2
        + 3 * a^3 * e1 * e2^2 * r3^2
        + 4 * a^3 * e1 * r3^2
        - 9 * a^3 * e2^3 * r3^2
        + 4 * a^3 * e2^2 * r3^2
        - 4 * a^3 * e2 * r3^2
        - 12 * a^2 * w * wbar * e1 * e2 * r3^3
        + 4 * a^2 * w * r3bar * e1 * e2 * r3^3
        - 9 * a^2 * w * e1^3 * r3^2
        - 3 * a^2 * w * e1^2 * e2 * r3^2
        - 3 * a^2 * w * e1 * e2^2 * r3^2
        + 12 * a^2 * w * e1 * e2 * r3^2
        - 9 * a^2 * w * e2^3 * r3^2
        - 3 * a^2 * e1^3 * r3^3
        + 3 * a^2 * e1^2 * e2 * r3^3
        + 3 * a^2 * e1 * e2^2 * r3^3
        - 4 * a^2 * e1 * e2 * r3^3
        - 3 * a^2 * e2^3 * r3^3
        + 3 * a * w * e1^3 * r3^3
        + 3 * a * w * e1^2 * e2 * r3^3
        - 3 * a * w * e1 * e2^2 * r3^3
        - 3 * a * w * e2^3 * r3^3
    )

    p_9 = (
        -3 * a^8 * w * wbar * r3^2
        + 3 * a^8 * w * r3bar * r3^2
        - 2 * a^8 * w * e1 * r3
        - 2 * a^8 * w * e2 * r3
        + 2 * a^8 * w * r3
        - a^8 * wbar * r3^3
        + a^8 * r3bar * r3^3
        + 2 * a^8 * e1 * r3^2
        + 2 * a^8 * e2 * r3^2
        - 2 * a^8 * r3^2
        + 3 * a^7 * w * e1 * r3^2
        - 3 * a^7 * w * e2 * r3^2
        + 3 * a^7 * wbar * e1 * r3^2
        - 3 * a^7 * wbar * e2 * r3^2
        - 3 * a^7 * r3bar * e1 * r3^2
        + 3 * a^7 * r3bar * e2 * r3^2
        + 2 * a^7 * e1^2 * r3
        + a^7 * e1 * r3^3
        - 2 * a^7 * e1 * r3
        - 2 * a^7 * e2^2 * r3
        - a^7 * e2 * r3^3
        + 2 * a^7 * e2 * r3
        + 3 * a^6 * w * wbar^3 * r3^2
        - 3 * a^6 * w * wbar^2 * r3bar * r3^2
        + 6 * a^6 * w * wbar^2 * e1 * r3
        + 6 * a^6 * w * wbar^2 * e2 * r3
        - 6 * a^6 * w * wbar^2 * r3
        - 4 * a^6 * w * wbar * r3bar * e1 * r3
        - 4 * a^6 * w * wbar * r3bar * e2 * r3
        + 4 * a^6 * w * wbar * r3bar * r3
        + 3 * a^6 * w * wbar * e1^2
        + 6 * a^6 * w * wbar * e1 * e2
        - 6 * a^6 * w * wbar * e1
        + 3 * a^6 * w * wbar * e2^2
        - 6 * a^6 * w * wbar * e2
        + 3 * a^6 * w * wbar
        - a^6 * w * r3bar * e1^2
        - 2 * a^6 * w * r3bar * e1 * e2
        + 2 * a^6 * w * r3bar * e1
        - a^6 * w * r3bar * e2^2
        + 2 * a^6 * w * r3bar * e2
        - a^6 * w * r3bar
        - a^6 * w * e1 * r3^3
        - a^6 * w * e2 * r3^3
        + a^6 * wbar^3 * r3^3
        - a^6 * wbar^2 * r3bar * r3^3
        + 3 * a^6 * wbar^2 * e1 * r3^2
        + 3 * a^6 * wbar^2 * e2 * r3^2
        - 5 * a^6 * wbar * r3bar * e1 * r3^2
        - 5 * a^6 * wbar * r3bar * e2 * r3^2
        + 2 * a^6 * wbar * r3bar * r3^2
        + 3 * a^6 * wbar * e1^2 * r3
        + 6 * a^6 * wbar * e1 * e2 * r3
        - 2 * a^6 * wbar * e1 * r3
        + 3 * a^6 * wbar * e2^2 * r3
        - 2 * a^6 * wbar * e2 * r3
        - a^6 * wbar * r3
        - 3 * a^6 * r3bar * e1^2 * r3
        - 6 * a^6 * r3bar * e1 * e2 * r3
        + 4 * a^6 * r3bar * e1 * r3
        - 3 * a^6 * r3bar * e2^2 * r3
        + 4 * a^6 * r3bar * e2 * r3
        - a^6 * r3bar * r3
        + a^6 * e1^3
        + 3 * a^6 * e1^2 * e2
        - 3 * a^6 * e1^2 * r3^2
        - 2 * a^6 * e1^2
        + 3 * a^6 * e1 * e2^2
        + 6 * a^6 * e1 * e2 * r3^2
        - 4 * a^6 * e1 * e2
        + a^6 * e1
        + a^6 * e2^3
        - 3 * a^6 * e2^2 * r3^2
        - 2 * a^6 * e2^2
        + a^6 * e2
        - 9 * a^5 * w * wbar^2 * e1 * r3^2
        + 9 * a^5 * w * wbar^2 * e2 * r3^2
        + 6 * a^5 * w * wbar * r3bar * e1 * r3^2
        - 6 * a^5 * w * wbar * r3bar * e2 * r3^2
        - 12 * a^5 * w * wbar * e1^2 * r3
        + 12 * a^5 * w * wbar * e1 * r3
        + 12 * a^5 * w * wbar * e2^2 * r3
        - 12 * a^5 * w * wbar * e2 * r3
        + 4 * a^5 * w * r3bar * e1^2 * r3
        - 4 * a^5 * w * r3bar * e1 * r3
        - 4 * a^5 * w * r3bar * e2^2 * r3
        + 4 * a^5 * w * r3bar * e2 * r3
        - 3 * a^5 * w * e1^3
        - 3 * a^5 * w * e1^2 * e2
        + 6 * a^5 * w * e1^2
        + 3 * a^5 * w * e1 * e2^2
        - 3 * a^5 * w * e1
        + 3 * a^5 * w * e2^3
        - 6 * a^5 * w * e2^2
        + 3 * a^5 * w * e2
        - 3 * a^5 * wbar^2 * e1 * r3^3
        + 3 * a^5 * wbar^2 * e2 * r3^3
        + 2 * a^5 * wbar * r3bar * e1 * r3^3
        - 2 * a^5 * wbar * r3bar * e2 * r3^3
        - 6 * a^5 * wbar * e1^2 * r3^2
        + 6 * a^5 * wbar * e2^2 * r3^2
        + 5 * a^5 * r3bar * e1^2 * r3^2
        - 2 * a^5 * r3bar * e1 * r3^2
        - 5 * a^5 * r3bar * e2^2 * r3^2
        + 2 * a^5 * r3bar * e2 * r3^2
        - 3 * a^5 * e1^3 * r3
        - 3 * a^5 * e1^2 * e2 * r3
        + a^5 * e1^2 * r3^3
        + 2 * a^5 * e1^2 * r3
        + 3 * a^5 * e1 * e2^2 * r3
        + a^5 * e1 * r3
        + 3 * a^5 * e2^3 * r3
        - a^5 * e2^2 * r3^3
        - 2 * a^5 * e2^2 * r3
        - a^5 * e2 * r3
        + 3 * a^4 * w * wbar^2 * e1 * r3^3
        + 3 * a^4 * w * wbar^2 * e2 * r3^3
        - 2 * a^4 * w * wbar * r3bar * e1 * r3^3
        - 2 * a^4 * w * wbar * r3bar * e2 * r3^3
        + 15 * a^4 * w * wbar * e1^2 * r3^2
        - 6 * a^4 * w * wbar * e1 * e2 * r3^2
        - 6 * a^4 * w * wbar * e1 * r3^2
        + 15 * a^4 * w * wbar * e2^2 * r3^2
        - 6 * a^4 * w * wbar * e2 * r3^2
        - 5 * a^4 * w * r3bar * e1^2 * r3^2
        + 2 * a^4 * w * r3bar * e1 * e2 * r3^2
        + 2 * a^4 * w * r3bar * e1 * r3^2
        - 5 * a^4 * w * r3bar * e2^2 * r3^2
        + 2 * a^4 * w * r3bar * e2 * r3^2
        + 9 * a^4 * w * e1^3 * r3
        + 3 * a^4 * w * e1^2 * e2 * r3
        - 12 * a^4 * w * e1^2 * r3
        + 3 * a^4 * w * e1 * e2^2 * r3
        + 3 * a^4 * w * e1 * r3
        + 9 * a^4 * w * e2^3 * r3
        - 12 * a^4 * w * e2^2 * r3
        + 3 * a^4 * w * e2 * r3
        + 3 * a^4 * wbar * e1^2 * r3^3
        - 6 * a^4 * wbar * e1 * e2 * r3^3
        + 2 * a^4 * wbar * e1 * r3^3
        + 3 * a^4 * wbar * e2^2 * r3^3
        + 2 * a^4 * wbar * e2 * r3^3
        - 2 * a^4 * r3bar * e1^2 * r3^3
        - 2 * a^4 * r3bar * e2^2 * r3^3
        + 3 * a^4 * e1^3 * r3^2
        - 3 * a^4 * e1^2 * e2 * r3^2
        + 2 * a^4 * e1^2 * r3^2
        - 3 * a^4 * e1 * e2^2 * r3^2
        + 4 * a^4 * e1 * e2 * r3^2
        - 2 * a^4 * e1 * r3^2
        + 3 * a^4 * e2^3 * r3^2
        + 2 * a^4 * e2^2 * r3^2
        - 2 * a^4 * e2 * r3^2
        - 6 * a^3 * w * wbar * e1^2 * r3^3
        + 6 * a^3 * w * wbar * e2^2 * r3^3
        + 2 * a^3 * w * r3bar * e1^2 * r3^3
        - 2 * a^3 * w * r3bar * e2^2 * r3^3
        - 9 * a^3 * w * e1^3 * r3^2
        + 3 * a^3 * w * e1^2 * e2 * r3^2
        + 6 * a^3 * w * e1^2 * r3^2
        - 3 * a^3 * w * e1 * e2^2 * r3^2
        + 9 * a^3 * w * e2^3 * r3^2
        - 6 * a^3 * w * e2^2 * r3^2
        - a^3 * e1^3 * r3^3
        + 3 * a^3 * e1^2 * e2 * r3^3
        - 2 * a^3 * e1^2 * r3^3
        - 3 * a^3 * e1 * e2^2 * r3^3
        + a^3 * e2^3 * r3^3
        + 2 * a^3 * e2^2 * r3^3
        + 3 * a^2 * w * e1^3 * r3^3
        - 3 * a^2 * w * e1^2 * e2 * r3^3
        - 3 * a^2 * w * e1 * e2^2 * r3^3
        + 3 * a^2 * w * e2^3 * r3^3
    )

    p_10 = (
        a^8 * w * wbar * r3^3
        - a^8 * w * r3bar * r3^3
        + a^8 * w * e1 * r3^2
        + a^8 * w * e2 * r3^2
        - a^8 * w * r3^2
        - a^8 * e1 * r3^3
        - a^8 * e2 * r3^3
        + a^8 * r3^3
        - a^7 * w * e1 * r3^3
        + a^7 * w * e2 * r3^3
        - a^7 * wbar * e1 * r3^3
        + a^7 * wbar * e2 * r3^3
        + a^7 * r3bar * e1 * r3^3
        - a^7 * r3bar * e2 * r3^3
        - a^7 * e1^2 * r3^2
        + a^7 * e1 * r3^2
        + a^7 * e2^2 * r3^2
        - a^7 * e2 * r3^2
        - a^6 * w * wbar^3 * r3^3
        + a^6 * w * wbar^2 * r3bar * r3^3
        - 3 * a^6 * w * wbar^2 * e1 * r3^2
        - 3 * a^6 * w * wbar^2 * e2 * r3^2
        + 3 * a^6 * w * wbar^2 * r3^2
        + 2 * a^6 * w * wbar * r3bar * e1 * r3^2
        + 2 * a^6 * w * wbar * r3bar * e2 * r3^2
        - 2 * a^6 * w * wbar * r3bar * r3^2
        - 3 * a^6 * w * wbar * e1^2 * r3
        - 6 * a^6 * w * wbar * e1 * e2 * r3
        + 6 * a^6 * w * wbar * e1 * r3
        - 3 * a^6 * w * wbar * e2^2 * r3
        + 6 * a^6 * w * wbar * e2 * r3
        - 3 * a^6 * w * wbar * r3
        + a^6 * w * r3bar * e1^2 * r3
        + 2 * a^6 * w * r3bar * e1 * e2 * r3
        - 2 * a^6 * w * r3bar * e1 * r3
        + a^6 * w * r3bar * e2^2 * r3
        - 2 * a^6 * w * r3bar * e2 * r3
        + a^6 * w * r3bar * r3
        - a^6 * w * e1^3
        - 3 * a^6 * w * e1^2 * e2
        + 3 * a^6 * w * e1^2
        - 3 * a^6 * w * e1 * e2^2
        + 6 * a^6 * w * e1 * e2
        - 3 * a^6 * w * e1
        - a^6 * w * e2^3
        + 3 * a^6 * w * e2^2
        - 3 * a^6 * w * e2
        + a^6 * w
        - a^6 * wbar^2 * r3^3
        + a^6 * wbar * r3bar * e1 * r3^3
        + a^6 * wbar * r3bar * e2 * r3^3
        - 2 * a^6 * wbar * e1 * r3^2
        - 2 * a^6 * wbar * e2 * r3^2
        + 2 * a^6 * wbar * r3^2
        + a^6 * r3bar * e1^2 * r3^2
        + 2 * a^6 * r3bar * e1 * e2 * r3^2
        - a^6 * r3bar * e1 * r3^2
        + a^6 * r3bar * e2^2 * r3^2
        - a^6 * r3bar * e2 * r3^2
        + a^6 * e1^2 * r3^3
        - a^6 * e1^2 * r3
        - 2 * a^6 * e1 * e2 * r3^3
        - 2 * a^6 * e1 * e2 * r3
        + 2 * a^6 * e1 * r3
        + a^6 * e2^2 * r3^3
        - a^6 * e2^2 * r3
        + 2 * a^6 * e2 * r3
        - a^6 * r3
        + 3 * a^5 * w * wbar^2 * e1 * r3^3
        - 3 * a^5 * w * wbar^2 * e2 * r3^3
        - 2 * a^5 * w * wbar * r3bar * e1 * r3^3
        + 2 * a^5 * w * wbar * r3bar * e2 * r3^3
        + 6 * a^5 * w * wbar * e1^2 * r3^2
        - 6 * a^5 * w * wbar * e1 * r3^2
        - 6 * a^5 * w * wbar * e2^2 * r3^2
        + 6 * a^5 * w * wbar * e2 * r3^2
        - 2 * a^5 * w * r3bar * e1^2 * r3^2
        + 2 * a^5 * w * r3bar * e1 * r3^2
        + 2 * a^5 * w * r3bar * e2^2 * r3^2
        - 2 * a^5 * w * r3bar * e2 * r3^2
        + 3 * a^5 * w * e1^3 * r3
        + 3 * a^5 * w * e1^2 * e2 * r3
        - 6 * a^5 * w * e1^2 * r3
        - 3 * a^5 * w * e1 * e2^2 * r3
        + 3 * a^5 * w * e1 * r3
        - 3 * a^5 * w * e2^3 * r3
        + 6 * a^5 * w * e2^2 * r3
        - 3 * a^5 * w * e2 * r3
        + 2 * a^5 * wbar * e1 * r3^3
        - 2 * a^5 * wbar * e2 * r3^3
        - a^5 * r3bar * e1^2 * r3^3
        + a^5 * r3bar * e2^2 * r3^3
        + 2 * a^5 * e1^2 * r3^2
        - 2 * a^5 * e1 * r3^2
        - 2 * a^5 * e2^2 * r3^2
        + 2 * a^5 * e2 * r3^2
        - 3 * a^4 * w * wbar * e1^2 * r3^3
        + 6 * a^4 * w * wbar * e1 * e2 * r3^3
        - 3 * a^4 * w * wbar * e2^2 * r3^3
        + a^4 * w * r3bar * e1^2 * r3^3
        - 2 * a^4 * w * r3bar * e1 * e2 * r3^3
        + a^4 * w * r3bar * e2^2 * r3^3
        - 3 * a^4 * w * e1^3 * r3^2
        + 3 * a^4 * w * e1^2 * e2 * r3^2
        + 3 * a^4 * w * e1^2 * r3^2
        + 3 * a^4 * w * e1 * e2^2 * r3^2
        - 6 * a^4 * w * e1 * e2 * r3^2
        - 3 * a^4 * w * e2^3 * r3^2
        + 3 * a^4 * w * e2^2 * r3^2
        - a^4 * e1^2 * r3^3
        + 2 * a^4 * e1 * e2 * r3^3
        - a^4 * e2^2 * r3^3
        + a^3 * w * e1^3 * r3^3
        - 3 * a^3 * w * e1^2 * e2 * r3^3
        + 3 * a^3 * w * e1 * e2^2 * r3^3
        - a^3 * w * e2^3 * r3^3
    )

    p = SVector{11, ComplexF64}(p_10, p_9, p_8, p_7, p_6, p_5, p_4, p_3, p_2, p_1, p_0)
    return p
end

"""
Compute the coefficients of 2*Nth order polynomial which defines the critical
curves for the binary lens case (N = 2).
"""
function poly_coeffs_critical_binary(ϕ::Float64, a::Float64, e1::Float64)
    p_0 = exp(-1im * ϕ)
    p_1 = 0. + 0im
    p_2 = -2 * a^2 * exp(-1im * ϕ) - 1.0
    p_3 = -4 * a * e1 + 2 * a
    p_4 = a^4 * exp(-1im * ϕ) - a^2

    p = SVector{5, ComplexF64}(p_4, p_3, p_2, p_1, p_0)
    return p
end

function poly_coeffs_critical_triple(ϕ::Float64, a::Float64, r3::ComplexF64, e1::Float64, e2::Float64)
    x = exp(-1im * ϕ)

    p_0 = x
    p_1 = -2 * x * r3
    p_2 = -2 * a^2 * x - 1 + x * r3^2
    p_3 = 4 * a^2 * x * r3 - 2 * a * e1 + 2 * a * e2 + 2 * e1 * r3 + 2 * e2 * r3
    p_4 = (
        a^4 * x
        - 3 * a^2 * e1
        - 3 * a^2 * e2
        + 2 * a^2
        - 2 * a^2 * x * r3^2
        + 4 * a * e1 * r3
        - 4 * a * e2 * r3
        - e1 * r3^2
        - e2 * r3^2
    )
    p_5 = (
        -2 * a^4 * x * r3
        + 2 * a^2 * e1 * r3
        + 2 * a^2 * e2 * r3
        - 2 * a * e1 * r3^2
        + 2 * a * e2 * r3^2
    )
    p_6 = (
        a^4 * e1
        + a^4 * e2
        - a^4
        + a^4 * x * r3^2
        - a^2 * e1 * r3^2
        - a^2 * e2 * r3^2
    )

    p = SVector{7, ComplexF64}(p_6, p_5, p_4, p_3, p_2, p_1, p_0)
    return p
end

function lens_eq(z::ComplexF64, params::Dict, nlenses::Int=2)
    zbar = conj(z)

    if nlenses == 2
        a, e1 = params["a"], params["e1"]
        return z - e1 / (zbar- a)- (1.0 - e1) / (zbar+ a)
    elseif nlenses == 3
        a, r3, e1, e2 = params["a"], params["r3"], params["e1"], params["e2"]
        return (
            z
           - e1 / (zbar - a)
           - e2 / (zbar + a)
           - (1.0- e1- e2) / (zbar- conj(r3))
        )
    else
        throw(ArgumentError("`nlenses` has to be set to be <= 3."))
    end
end

function lens_eq_det_jac(z::ComplexF64, params::Dict, nlenses::Int=2)
    zbar = conj(z)

    if nlenses == 1
        return 1.0- 1.0 / abs(zbar^2)

    elseif nlenses == 2
        a, e1 = params["a"], params["e1"]
        return 1.0- abs(e1 / (zbar - a) ^ 2+ (1.0 - e1) / (zbar + a) ^ 2) ^ 2

    elseif nlenses == 3
        a, r3, e1, e2 = params["a"], params["r3"], params["e1"], params["e2"]
        return (
            1.0
           - abs(
                e1 / (zbar- a) ^ 2
               + e2 / (zbar+ a) ^ 2
               + (1.0- e1- e2) / (zbar- conj(r3)) ^ 2
            )
            ^ 2
        )
    else
        throw(ArgumentError("`nlenses` has to be set to be <= 3."))
    end
end

function critical_and_caustic_curves(params::Dict, nlenses::Int=2, npts::Int=400)
    ϕ = collect(LinRange(-π, π, npts))
    coeffs = zeros(ComplexF64, npts, 2*nlenses + 1)
    z_cr = zeros(ComplexF64, npts, 2*nlenses)

    if nlenses == 2
        s, q = params["s"], params["q"]
        a, e1, e2 = 0.5*s, q/(1 + q), 1. - q/(1 + q)
        _params = Dict("a" => a, "e1" => e1)
    
        for i in 1:npts
            coeffs[i, :] = poly_coeffs_critical_binary(ϕ[i], a, e1)
            z_cr[i, :] = roots(coeffs[i, :], polish=true)
        end
    elseif nlenses == 3
        s, q, q3, r3, psi = params["s"], params["q"], params["q3"], params["r3"], params["psi"]
        a, e1, e2 = 0.5*s, q/(1 + q + q3), 1/(1 + q + q3)
        r3 = r3*exp(1im*psi) - a
        _params = Dict("a" => a, "r3" => r3, "e1" => e1, "e2" => e2)

        for i in 1:npts
            coeffs[i, :] = poly_coeffs_critical_triple(ϕ[i], a, r3, e1, e2)
            z_cr[i, :] = roots(coeffs[i, :], polish=true)
        end
    else
        throw(ArgumentError("`nlenses` has to be set to be <= 3."))
    end

    # Permute roots so that they form contiguous curves
    for i in 1:length(z_cr[:, 1]) - 1
        idcs = match_points(z_cr[i, :], z_cr[i+1, :])
        z_cr[i+1, :] = z_cr[i+1, idcs]
    end

    # Caustics are critical curves mapped by the lens equation
    z_ca = reshape(lens_eq.(z_cr[:], Ref(_params), Ref(nlenses)), size(z_cr))

    # Shift by the centre of mass
    x_cm = 0.5*s*(1-q)/(1 + q)
    z_cr, z_ca = z_cr, z_ca .+ x_cm

    return z_cr, z_ca
end

"""
Compute the point source images of a binary or triple lens system 
at location `w` in the source plane.
"""
function images_point_source(
    w::ComplexF64, 
    params::Dict, 
    nlenses::Int=2, 
    roots_init::Union{Vector{ComplexF64},Nothing}=nothing
)
    if nlenses == 2
        a, e1 = params["a"], params["e1"]
        # Compute complex polynomial coefficients for each element of w
        coeffs = poly_coeffs_binary(w, a, e1)
        if roots_init == nothing
            # Compute the roots of the polynomial
            z = roots(Vector(coeffs), polish=true)
        else
            z = roots(Vector(coeffs), Vector(roots_init), polish=true)
        end
        z_mask = SVector{5, Bool}(abs(lens_eq(z[i], params, nlenses) - w) .< 1e-06 for i in 1:5)

    elseif nlenses == 3
        a, r3, e1, e2 = params["a"], params["r3"], params["e1"], params["e2"]
        # Compute complex polynomial coefficients for each element of w
        coeffs = poly_coeffs_triple(w, a, r3, e1, e2)
        if roots_init == nothing
            z = roots(Vector(coeffs), polish=true)
        else
            z = roots(Vector(coeffs), Vector(roots_init), polish=true)
        end
        z_mask = SVector{10, Bool}(abs(lens_eq(z[i], params, nlenses) - w) .< 1e-06 for i in 1:10)
    
    else
        throw(ArgumentError("`nlenses` has to be set to be <= 3."))
    end

    z_parity = sign.(lens_eq_det_jac.(z, Ref(params), Ref(nlenses)))

    return z, z_mask, z_parity
end

"""
mag_point_source(w::ComplexF64, params::Dict, nlenses=2)

Compute the magnification of a point source for a system with `nlenses`
lenses. If `nlenses` is 2 (binary lens) or 3 (triple lens), the coordinate
system is set up such that the the origin is at the center of mass of the
first two lenses which are both located on the real line. The location of
the first lens is `-sq/(1 + q)`` and the second lens is at `s/(1 + q)`. The
optional third lens is located at an arbitrary position in the complex plane
`r_3e^{-ipsi}`. 

If `nlenses` is 2 only the parameters `s` and `q` should be specified. If
`nlenses` is 3, the parameters `s`, `q`, `q3`, `r3` and `psi` should be
specified.

### Input 

- `w::ComplexF64` -- Source position in the complex plane.
- `params::Dict` -- Dictionary of lens parameters.
- `nlenses::Int=2` -- Number of lenses in the system.

### Returns

- `mag::Float64` -- The point source magnification evaluated at w.
"""
function mag_point_source(w::ComplexF64, params::Dict, nlenses=2)
    if nlenses == 2
        s, q = params["s"], params["q"]
        a, e1, e2 = 0.5*s, q/(1 + q), 1. - q/(1 + q)
        _params = Dict("a" => a, "e1" => e1)

        # Shift w by x_cm
        x_cm = 0.5*s*(1-q)/(1 + q)
        w -= x_cm
    elseif nlenses == 3
        s, q, q3, r3, psi = params["s"], params["q"], params["q3"], params["r3"], params["psi"]
        a, e1, e2 = 0.5*s, q/(1 + q + q3), 1/(1 + q + q3)
        r3 = r3*exp(1im*psi) - a
        _params = Dict("a" => a, "r3" => r3, "e1" => e1, "e2" => e2)

        # Shift w by x_cm
        x_cm = 0.5*s*(1-q)/(1+q)
        w -= x_cm
    else
        throw(ArgumentError("`nlenses` has to be set to be <= 3."))
    end

    z, z_mask, z_parity = images_point_source(w, _params, nlenses)
    det = lens_eq_det_jac.(z, Ref(_params), Ref(nlenses))
    mag = (1.0./abs.(det)).*z_mask
    return sum(mag)
end

end