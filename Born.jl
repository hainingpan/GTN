function sample_Ω(k::Int, params::Dict{String,Any}, psi::BlockMatrix{T}) where {T}
    χa = params["χa"]
    χp = params["χp"]
    a1 = params["a1"]
    a2 = params["a2"]
    b1 = params["b1"]
    b2 = params["b2"]
    L = params["L"]
    if k<L
        psi_contract = [Meta.parse("idx$k"), Meta.parse("idx$(k+1)")]
        T_contract = [Meta.parse("leftd$k"), Meta.parse("rightd$k")]
    else
        psi_contract = [Meta.parse("idx$L"), Meta.parse("idx1")]
        T_contract = [Meta.parse("leftd$L"), Meta.parse("rightd$L")]
    end
    i_names = map(s -> Meta.parse(s), ["leftd$k", "rightd$k", "leftu$k", "rightu$k"])
    Ω = BlockMatrix(i_names, [χa, χa, χp, χp])

    g = psi[psi_contract,psi_contract][1,2]
    if mod(k,2)==1
        if g==0
            r = (b1-a1)/(b1-a1+b2-a2)
            if rand()>r
                x = a2+(1-rand())*(b2-a2)
            else
                x = -b1+(1-rand())*(b1-a1)
            end
        else
            function Finv(x)
                λ1 = (b2+a2)/((b1-a1)*(a1+b1+a2+b2))
                λ2 = (b1+a1)/((b2-a2)*(a1+b1+a2+b2))
                v1 = λ1* (b1 + g/2*b1^2)
                v2 = λ2* (-a2 + g/2*a2^2)
                F1 = λ1*(-a1+b1-g/2*(a1^2-b1^2))    
                if x <=  F1
                    return (1-(1+2*g*(v1-x)/λ1)^0.5)/g
                else
                    return (1-(1+2*g*(v2-x+F1)/λ2)^0.5)/g
                end
            end
            x = Finv(rand())
        end
        ϕ = rand()*2*pi  
        n1 = x
        n2 = sqrt(1-x^2)*cos(ϕ)
        n3 = sqrt(1-x^2)*sin(ϕ)
    else
#         ϕ = rand()*pi
#         n1 = 0
#         n2 = sin(ϕ)
#         n3 = cos(ϕ)
        r = (b1-a1)/(b1-a1+b2-a2)
        if rand()>r
            n20 = a2+(1-rand())*(b2-a2)
        else
            n20 = -b1+(1-rand())*(b1-a1)
        end
        n10 = (1-n20^2)^0.5
        ϕ0 = rand()*2*pi
        p0 = 1-g*n10*sin(ϕ0)

        N_mc = params["N_mc"]
        acc_ratio = 0
        for i_mc in 1:N_mc
            r = (b1-a1)/(b1-a1+b2-a2)
            if rand()>r
                n21 = a2+(1-rand())*(b2-a2)
            else
                n21 = -b1+(1-rand())*(b1-a1)
            end
            n11 = (1-n21^2)^0.5
            p1 = 1-g*n11*sin(ϕ0)

            if rand() ≤ abs(p1/p0)
                acc_ratio += 1
                n10 = n11
                n20 = n21
                p0 = p1
            end

            ϕ1 = rand()*2*pi
            p1 = 1-g*n10*sin(ϕ1)
            if rand() ≤ abs(p1/p0)
                acc_ratio += 1
                ϕ0 = ϕ1
                p0 = p1
            end
        end
        n1 = n10*sin(ϕ0)
        n2 = n20
        n3 = n10*cos(ϕ0)
    end
    Ω[i_names,i_names] = [0 n1 n2 n3;-n1 0 -n3 n2;-n2 n3 0 -n1;-n3 -n2 n1 0]
    psi = schurContract(psi, Ω, psi_contract, T_contract)
    return psi
end