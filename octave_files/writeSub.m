function [] = writeSub(p, outfile, train=false)
%Write predictions to outfile
	%if train set to True then use indices starting at 0
	%otherwise, assume test data (and in order of PassengerId)

fid = fopen(outfile, 'w');
fdisp(fid, 'PassengerId,Survived');
fclose(fid);

len = length(p)-1;
indices = 892:(892+len);
if train
	indices = 1:(1+len);
end

csvwrite(outfile, [indices' p],'-append');
end
