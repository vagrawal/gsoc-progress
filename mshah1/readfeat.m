function y = readfeat(cepfile, dim, maxtoread)
%Y = READFEAT(CEPFILE, [DIM], [MAXVECSTOREAD])
%Reads cepfile in appropriate byte order
%If dim is not given, it returns a 1-d array; else it returns a matrix
%if maxvecstoread is not given, it reads the entire file

fileinfo = dir(cepfile);
numfloat = (fileinfo.bytes / 4) - 1;

f = fopen(cepfile,'r','ieee-le');
n = fread(f,1,'int32');
if (n ~= numfloat)
   fclose(f);
   f = fopen(cepfile,'r','ieee-be');
   n = fread(f,1,'int32');
   if (n ~= numfloat)
       fclose(f);
       y =  [];
       return;
   end
end

if (nargin  == 3) % Read only requested number of vectors
    [y, numread] = fread(f, maxtoread*dim, 'float32');
else
    [y, numread] = fread(f,'float32');
end

if (nargin >= 2)
    y = reshape(y,dim,numread/dim);
end

fclose(f);
