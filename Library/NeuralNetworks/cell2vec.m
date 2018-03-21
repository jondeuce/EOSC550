function a = cell2vec(A)
%a = cell2vec(A)
%

a = [];
for i=1:length(A)
    a = [a ; A{i}(:)];
end