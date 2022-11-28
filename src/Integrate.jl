module Integrate

using QuadGK

include("PointSource.jl")
import .PointSource: lens_eq 

export integrate_unif, integrate_ld

function trapz(x, y)
    I = 0.
    for i in 1:length(x)-1
        I += (x[i+1] - x[i]) * (y[i+1] + y[i])
    end
    return I
end

function integrate_unif(contour)
    # Integrate using trapezoidal rule
    I1 = trapz(imag(contour.z), 0.5*real(contour.z))
    I2 = trapz(real(contour.z), -0.5*imag(contour.z))
    return I1 + I2
end


function brightness_profile(z, ρ, w0, params, u1=0.0, nlenses=2)
    w = lens_eq(z, params, nlenses)
    r = abs(w - w0) / ρ

    # See Dominik 1998 for details
    if r <= 1.
        B_r = 1. + sqrt(1. - r^2)
    else
        B_r = 1. - sqrt(1. - 1. / r^2)
    end
    I = 3.0 / (3.0 - u1) * (u1 * B_r + 1.0 - 2.0 * u1)
    return I
end


function integrate_ld(contour, w0, ρ, params, u1=0., nlenses=2, rtol=1e-3)
    z = contour.z
    
    # Compute the centroid of the contour 
    z0 = sum(z)/length(z)

    function P_integrand(y0, xl, yl)
        f(y) = brightness_profile(xl +1im*y, ρ, w0, params, u1, nlenses)
        I , _ = quadgk(f, y0, yl, rtol=rtol)
        return -0.5*I
    end

    function Q_integrand(x0, xl, yl)
        f(x) = brightness_profile(x +1im*yl, ρ, w0, params, u1, nlenses)
        I , _ = quadgk(f, x0, xl, rtol=rtol)
        return 0.5*I
    end

   # Evaluate the P and Q integrands for each point in the contour  
   P_integrands = zeros(length(z))
   Q_integrands = zeros(length(z))

    for i in 1:length(z)
         xl, yl = real(z[i]), imag(z[i])
         x0, y0 = real(z0), imag(z0)
         P_integrands[i] = P_integrand(y0, xl, yl)
         Q_integrands[i] = Q_integrand(x0, xl, yl)
    end

    # Compute the final integrals using the trapezoidal rule
    P = trapz(real(z), P_integrands)
    Q = trapz(imag(z), Q_integrands)
    return P + Q
end

end
