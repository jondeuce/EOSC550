classdef opZero
    % zero operator
    
    properties
        m
        n
        Amv
        ATmv
    end
    
    methods
        function this = opZero(m,n)
            this.m = m;
            this.n = n;
            this.Amv = @(x) zeros(m,size(x,2));
            this.ATmv = @(x) zeros(n,size(x,2));
        end
        function z = mtimes(this,x)
            z = this.Amv(x);
        end
        function this = ctranspose(this)
            temp   = this.m;
            this.m = this.n;
            this.n = temp;
            temp   = this.Amv;
            this.Amv = this.ATmv;
            this.ATmv = temp;
        end
 

        
        function this = convertGPUorPrecision(this,useGPU,precision)
            % do nothing
        end

    end
end

