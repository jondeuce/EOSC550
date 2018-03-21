classdef opEye 
    % identity operator
    
    properties
        m
        n
        Amv
        ATmv
    end
    
    methods
        function this = opEye(n)
            this.m = n;
            this.n = n;
            this.Amv = @(x) x;
            this.ATmv = @(x) x;
        end
        function z = mtimes(this,x)
            z = this.Amv(x);
        end
        
        function this = convertGPUorPrecision(this,useGPU,precision)
            % do nothing
        end

    end
end

