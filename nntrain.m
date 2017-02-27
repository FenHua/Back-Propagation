function[nn,L]=nntrain(nn,train_x,train_y,opts)
m=size(train_x,1);
batchsize=opts.batchsize;
numepochs=opts.numepochs;
numbatchs=floor(m/batchsize);
L=zeros(numepochs*numbatchs,1);
n=1;
for k=1:numepochs
    tic
    kk=randperm(m);
    for j=1:numbatchs
        batch_x=train_x(kk((j-1)*batchsize+1:j*batchsize),:);
        batch_y=train_y(kk((j-1)*batchsize+1:j*batchsize),:);
        nn=nnff(nn,batch_x,batch_y);
        nn=nnbp(nn);
        nn=nngrads(nn);
        L(n)=nn.loss;
        n=n+1;
    end
    t=toc;%function reads the elapsed time from the stopwatch timer started by the tic function.
    nn=nnff(nn,train_x,train_y);
    str_perf=sprintf('; Full-batch train err=%f',nn.loss);
    disp(['NN train:epoch ' num2str(k) '/' num2str(opts.numepochs) '.Took' num2str(t) 'second' '.Mini-batch mean squared srror on training set is ' num2str(mean(L((n-numbatchs):(n-1)))) str_perf]);
end
end
function nn=nnff(nn,x,y)
n=nn.n;
m=size(x,1);
x=[ones(m,1) x];
nn.a{1}=x;
%forward feedback pass
for i=2:n-1
    nn.a{i}=sigm(nn.a{i-1}*nn.W{i-1}');
    % add the basis term
    nn.a{i}=[ones(m,1) nn.a{i}];
end
nn.a{n}=sigm(nn.a{n-1}*nn.W{n-1}');
%error and loss
nn.error=y-nn.a{n};
nn.loss=1/2 *sum(sum(nn.error.^2))/m;
end

function X=sigm(P)
X=1./(1+exp(-P));
end

function nn=nnbp(nn)
n=nn.n;
d{n}=-nn.error.*(nn.a{n}.*(1-nn.a{n}));
for i=(n-1):-1:2
    %derivative of the activation function
    d_act=nn.a{i}.*(1-nn.a{i});
    %backpropagate first derivatives
    if i+1==n 
        %in this case in d{n} there is not the bias term to be removed
        d{i}=(d{i+1}*nn.W{i}).*d_act;
    else
        % in this case in d{i} the bias term has to be removed
        d{i}=(d{i+1}(:,2:end)*nn.W{i}).*d_act;
    end
end
for i=1:(n-1)
    if i+1==n
        nn.dW{i}=(d{i+1}'*nn.a{i})/size(d{i+1},1);
    else
        nn.dW{i}=(d{i+1}(:,2:end)'*nn.a{i})/size(d{i+1},1);
    end
end
end

function nn=nngrads(nn)
for i=1:(nn.n-1)
    dW=nn.dW{i};
    dW=nn.learning_rate*dW;
    if(nn.momentum>0)
        nn.vW{i}=nn.momentum*nn.vW{i}+dW;
        dW=nn.vW{i};
    end
    nn.W{i}=nn.W{i}-dW;
end
end
        
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    