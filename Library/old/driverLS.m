% testing linear regression
clear 
close all

a = .001;
Y = [0 1+a; 1  1+2*a ; 1  1+3*a];
%Y = randn(3,2);

w = [10; 60];

c = Y*w + randn(3,1)*1e0;

F = @(x) 0.5*norm(Y*x-c)^2;

n = 65;
f = zeros(n,n);
t = linspace(-150,150,n);
for i=1:n
    for j=1:n
        x = [t(i);t(j)];
        f(i,j) = F(x);
    end
end

contourf(t,t,f,50)
w0 = [-100;100];
[w,rho,eta,W] = steepestDescent(Y,c,20,w0);
hold on
plot(W(1,:),W(2,:),'.r','markersize',15)
plot(W(1,:),W(2,:),'r')
hold off



if 1
    [w,rho,eta,W] = cgls(Y,c,3,w0);
    hold on
    plot(W(1,:),W(2,:),'.g','markersize',15)
    plot(W(1,:),W(2,:),'g')
    hold off
end
    