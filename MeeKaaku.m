


******************* MeeKaaku Face Recognition ****************************
% NB: Any work arising from the use of this code must provide due credit and cite the following work of the authors. 
% Ali Elmahmudi and Hassan Ugail, Deep face recognition using imperfect facial data, Future Generation Computer %Systems, 99, 213-225, 2019.

% MeeKaaku Face Recognition Code

% Install and compile MatConvNet by using the link below
untar('http://www.vlfeat.org/matconvnet/download/matconvnet-1.0-beta25.tar.gz') ;
cd matconvnet-1.0-beta25
run matlab/vl_compilenn ;
% Setup MatConvNet
run matlab/vl_setupnn ;

% Download pre-trained model from this website
% http://www.vlfeat.org/matconvnet/pretrained/

% Load the pre-trained model
net = load('vgg-face.mat') ;
net = vl_simplenn_tidy(net) ;
no_items=34;

% Select a training path
TrainingPath=uigetdir('..\','Select path for the Training set');
TrainingPath =strcat(TrainingPath,'\');
[X,Y,~]=FeaturesExtraction(TrainingPath,net,no_items);

% Select a testing path
TestPath=uigetdir('..\','Select path for the Testing set');
TestPath =strcat(TestPath,'\');
[X_test,Y_test,~]=FeaturesExtraction(TestPath,net,no_items);

% Training the classifiers
h = busydlg('Please wait while training all classifiers...','Training phase');
 try

    LinearClassifier = fitcecoc(X,Y); 
           
    t = templateSVM('Standardize',1,'KernelFunction','polynomial', 'KernelScale', 'auto', 'PolynomialOrder', 1.5);
    polynomialClassifier = fitcecoc(X, Y,'Learners',t,'FitPosterior',1);
           
    t2 = templateSVM('Standardize',1,'KernelFunction','gaussian', 'KernelScale', 'auto');
    KernelClassifier = fitcecoc(X,Y,'Learners',t2);
              
  delete(h);
  msgbox('!!! Training process complete !!! ');
catch
  delete(h);
 end
 % End of training stage for SVMs
 
 % Testing phase
 
 no_items=size(X_test,1);
 cou_rbf=0; cou_lsvm=0;cou_cs=0;cou_psvm=0;
   rbf=0;linear=0;pol=0;cs=0;        
    for cl_im=1:no_items
 
        %%% RBF SVMs
        tic
        [label_rbf, ~,~]=predict(KernelClassifier,X_test(cl_im,:));
        toc
        rbf=rbf+toc;
        if strcmp(label_rbf,Y_test(cl_im))==1
           cou_rbf=cou_rbf+1;
        end
        
        %%% Lieaner SVMs
        tic
        [label_SVML, ~,~]=predict(LinearClassifier,X_test(cl_im,:));
        toc
        linear=linear+toc;
        if strcmp(label_SVML,Y_test(cl_im))==1
            cou_lsvm=cou_lsvm+1;
        end
        
        %%% Pol SVMs
        tic
        [label_pol, ~,~]=predict(polynomialClassifier,X_test(cl_im,:));
        toc
        pol=pol+toc;
        if strcmp(label_pol,Y_test(cl_im))==1
           cou_psvm=cou_psvm+1;
        end
        
        %%% CS
        tic
        cos_res=cose_s(X,Y,X_test(cl_im,:));
        toc
        cs=cs+toc;
        if strcmp(cos_res,Y_test(cl_im))==1
           cou_cs=cou_cs+1;
        end

    end
   % Results of classification
   Percentage_rbf=(cou_rbf/no_items)*100;
   Percentage_pol=(cou_psvm/no_items)*100;
   Percentage_cs=(cou_cs/no_items)*100;
   Percentage_lsvm=(cou_lsvm/noi)*100;
********************************************************
*********** Feature extraction Function ****************
function [X,Y,Z]=FeaturesExtraction(Dir,net,no)

S = dir(Dir);
c=sum([S.isdir]);
X=[];  
Y={};
Z={};
N=numberofimages(Dir);
str1=strcat('Please wait...You have: ',num2str(N),' images');
h=waitbar(0,str1,'Name','Training Feature extraction'); 
o=0;
for i=3:c
    Dir2=strcat(Dir,S(i).name);
    imgs = dir(fullfile(Dir2, '*.jpg'));
    %%%%
    % Convert all images 2D into 1D
    for j=1:length(imgs)
        o=o+1;
       im = imread(fullfile(Dir2, imgs(j).name));
       z= imgs(j).name;
       %convert image from grayscale into rgb
       if length(size(im))<3
         im=cat(3,im,im,im);
       end
       im=single(im);
       %Resize image to 224*224
       [ro, co,~] = size(im);
       if (ro~=224)|| (co~=224)
          im=imresize(im, net.meta.normalization.imageSize(1:2));
       
       end
       
       im=im - net.meta.normalization.averageImage ;
     
       %Run the CNN
       res = vl_simplenn(net, im) ;
       feature = squeeze(gather(res(no).x));
       feature = feature(:);
       %add features to matrix X and their label to Y
       X = [X; feature'];
       y = S(i).name;
       Y = [Y; y];
       Z = [Z; z];
       waitbar(o/N,h);
    end
end
 delete(h);
 end 
*********************************************************************
********* Cosine Similarity *****************************************

function [predicted_result]=cose_s(c1,lab1,c2)
                               

 for i=1:size(c2,1)
    min1 = pdist2((c1(1,:)),(c2(i,:)),'cosine');
    ind=1;
    for j=1:size(c1,1)
       D = pdist2((c1(j,:)),(c2(i,:)),'cosine');
       if D<min1
           ind=j;
           min1=D;
       end
    end
       predicted_result=lab1(ind);
  end
********************************************************************
