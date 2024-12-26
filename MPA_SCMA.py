# (Available Codewords=4)
def MPA_SCMA_4(y,F,codebook,dv,df,h,noise_power,B,iterations):
    n_users = codebook.shape[0]
    n_REs = codebook.shape[2]
    n_cw = codebook.shape[1]
    N = y.shape[0]
    LLR = np.zeros((B,N));
    ind_dv = np.zeros((n_users,dv),dtype =np.int64);
    ind_df = np.zeros((n_REs,df),dtype=np.int64);
    N0=1./(noise_power)
    # for k in range(0,n_REs):
    #     RE_users = np.where(F[k] == 1)
    #     ind_df[k]=RE_users[0]
    # for j in range(0,n_users):
    #     user_REs = np.where(np.transpose(F)[j] == 1)
    #     ind_dv[j]=user_REs[0]
    ind_df = np.array([np.where(F[k] == 1)[0] for k in range(n_REs)])
    ind_dv = np.array([np.where(np.transpose(F)[j] == 1)[0] for j in range(n_users)])
    
    
    for n in range(0,N):
        #Step 1 : Initialization
        f = np.zeros((n_REs,n_cw,n_cw,n_cw));
        for k in range(0,n_REs):
            for m1 in range(0,n_cw):
                for m2 in range(0,n_cw):
                    for m3 in range(0,n_cw):
                        sum_ = np.zeros(2,dtype=np.float64)
                        sum_[0] = codebook[ind_df[k,0],m1,k,0]+codebook[ind_df[k,1],m2,k,0]+codebook[ind_df[k,2],m3,k,0]
                        sum_[1] = codebook[ind_df[k,0],m1,k,1]+codebook[ind_df[k,1],m2,k,1]+codebook[ind_df[k,2],m3,k,1]
                        distance = y[n,2*k] - sum_[0]+ 1j*(y[n,2*k+1]-sum_[1])
                        f[k,m1,m2,m3]=np.exp(-N0*((np.real(distance)**2+np.imag(distance)**2)**0.5))
        Iru = np.zeros((n_REs,n_users,n_cw));
        Iur = np.full((n_users,n_REs,n_cw),(1./n_cw))
        #Step 2 : Iterative Procedure
        for t in range(0,iterations):
            #update the Resource Node 
            for k in range(0,n_REs):
                for m1 in range(0,n_cw):
                    sIkj = np.zeros((n_cw*n_cw),dtype=np.float64);
                    for m2 in range(0,n_cw):
                        for m3 in range(0,n_cw):
                            sIkj[m2*n_cw+m3]=f[k,m1,m2,m3]*Iur[ind_df[k,1],k,m2]*Iur[ind_df[k,2],k,m3]
                    Iru[k,ind_df[k,0],m1]=np.sum(sIkj)
                for m2 in range(0,n_cw):
                    sIkj = np.zeros((n_cw*n_cw),dtype=np.float64);
                    for m1 in range(0,n_cw):
                        for m3 in range(0,n_cw):
                            sIkj[m1*n_cw+m3]=f[k,m1,m2,m3]*Iur[ind_df[k,0],k,m1]*Iur[ind_df[k,2],k,m3]
                    Iru[k,ind_df[k,1],m2]=np.sum(sIkj)
                for m3 in range(0,n_cw):
                    sIkj = np.zeros((n_cw*n_cw),dtype=np.float64);
                    for m1 in range(0,n_cw):
                        for m2 in range(0,n_cw):
                            sIkj[m1*n_cw+m2]=f[k,m1,m2,m3]*Iur[ind_df[k,0],k,m1]*Iur[ind_df[k,1],k,m2]
                    Iru[k,ind_df[k,2],m3]=np.sum(sIkj)
            #update Variable Node
            for j in range(0,n_users):
                for m in range(0,n_cw):
                    Iur[j,ind_dv[j,0],m]=(1/df)*Iru[ind_dv[j,1],j,m]
                    Iur[j,ind_dv[j,1],m]=(1/df)*Iru[ind_dv[j,0],j,m]
        #Step 3: LLR calculation
        I = np.zeros((n_cw,n_users));
        for j in range(0,n_users):
            for m in range(0,n_cw):
                I[m,j]=(1./n_cw)*Iur[j,ind_dv[j,0],m]*Iur[j,ind_dv[j,1],m]
        for j in range(0,n_users):
            LLR[2*j,n]=np.log((I[0,j]+I[1,j])/(I[2,j]+I[3,j]))
            LLR[2*j+1,n]=np.log((I[0,j]+I[2,j])/(I[1,j]+I[3,j]))
    return LLR
####################SCMA MPA (Available Codewords=2)################################################################
def MPA_SCMA_2(y,F,codebook,dv,df,h,noise_power,B,iterations):
    n_users = codebook.shape[0]
    n_REs = codebook.shape[2]
    n_cw = codebook.shape[1]
    N = y.shape[0]
    LLR = np.zeros((B,N))
    ind_dv = np.zeros((n_users,dv),dtype =np.int64)
    ind_df = np.zeros((n_REs,df),dtype=np.int64)
    N0=1./(noise_power)
    for k in range(0,n_REs):
        RE_users = np.where(F[k] == 1)
        ind_df[k] = RE_users[0]
    for j in range(0,n_users):
        user_REs = np.where(np.transpose(F)[j] == 1)
        ind_dv[j] = user_REs[0]
        
    for n in range(0,N):
        #Step 1 : Initialization
        f = np.zeros((n_REs,n_cw,n_cw,n_cw))
        for k in range(0,n_REs):
            for m1 in range(0,n_cw):
                for m2 in range(0,n_cw):
                    for m3 in range(0,n_cw):
                        sum_ = np.zeros(2,dtype=np.float64)
                        sum_[0] = codebook[ind_df[k,0],m1,k,0]+codebook[ind_df[k,1],m2,k,0]+codebook[ind_df[k,2],m3,k,0]
                        sum_[1] = codebook[ind_df[k,0],m1,k,1]+codebook[ind_df[k,1],m2,k,1]+codebook[ind_df[k,2],m3,k,1]
                        distance = y[n,2*k] - sum_[0]+ 1j*(y[n,2*k+1]-sum_[1])
                        f[k,m1,m2,m3] = np.exp(-N0*((np.real(distance)**2+np.imag(distance)**2)**0.5))
        Iru = np.zeros((n_REs,n_users,n_cw))
        Iur = np.full((n_users,n_REs,n_cw),(1./n_cw))
        #Step 2 : Iterative Procedure (Message passing between Resource and Variable Nodes)
        for t in range(0,iterations):
            #update the Resource Node 
            for k in range(0,n_REs):
                for m1 in range(0,n_cw):
                    sIkj = np.zeros((n_cw*n_cw),dtype=np.float64)
                    for m2 in range(0,n_cw):
                        for m3 in range(0,n_cw):
                            sIkj[m2*n_cw+m3] = f[k,m1,m2,m3]*Iur[ind_df[k,1],k,m2]*Iur[ind_df[k,2],k,m3]
                    Iru[k,ind_df[k,0],m1] = np.sum(sIkj)
                for m2 in range(0,n_cw):
                    sIkj = np.zeros((n_cw*n_cw),dtype=np.float64)
                    for m1 in range(0,n_cw):
                        for m3 in range(0,n_cw):
                            sIkj[m1*n_cw+m3] = f[k,m1,m2,m3]*Iur[ind_df[k,0],k,m1]*Iur[ind_df[k,2],k,m3]
                    Iru[k,ind_df[k,1],m2] = np.sum(sIkj)
                for m3 in range(0,n_cw):
                    sIkj = np.zeros((n_cw*n_cw),dtype=np.float64)
                    for m1 in range(0,n_cw):
                        for m2 in range(0,n_cw):
                            sIkj[m1*n_cw+m2] = f[k,m1,m2,m3]*Iur[ind_df[k,0],k,m1]*Iur[ind_df[k,1],k,m2]
                    Iru[k,ind_df[k,2],m3] = np.sum(sIkj) 
            #update Variable Node
            for j in range(0,n_users):
                for m in range(0,n_cw):
                    Iur[j,ind_dv[j,0],m] = (1/df)*Iru[ind_dv[j,1],j,m] 
                    Iur[j,ind_dv[j,1],m] = (1/df)*Iru[ind_dv[j,0],j,m] 
        #Step 3: LLR calculation
        I = np.zeros((n_cw,n_users))
        for j in range(0,n_users):
            for m in range(0,n_cw):
                I[m,j] = (1./n_cw)*Iur[j,ind_dv[j,0],m]*Iur[j,ind_dv[j,1],m]
        for j in range(0,n_users):
            LLR[j,n] = np.log(I[0,j]/I[1,j])
    return LLR