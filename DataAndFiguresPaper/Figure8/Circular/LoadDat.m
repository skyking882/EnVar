function [dat1]=LoadDat(fileToRead1,int,fileToReadA,m,n)

    dat1=importfiledata(fileToRead1,n);

    if(int==0)
        A = FFmatrix_fread(fileToReadA);
        dat1=A*dat1;
    end

end 