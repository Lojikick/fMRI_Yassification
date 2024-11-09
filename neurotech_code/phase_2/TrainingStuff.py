from sklearn import linear_model

X_frm = None
Y_img = None
Y_cap= None
reg = linear_model.Ridge(alpha=0.5) #alpha = lambda in RR
r1 = reg.fit(X_frm, Y_cap)
r2 = reg.fit(X_frm, Y_img)

#send r1 and r2 to testing phase
