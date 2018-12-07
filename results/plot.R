inp=read.table("tf.dat",header=T)

pdf("KDE.compare_bw_methods.pdf")
# par(lwd=4)

bw.method = c("nrd0","nrd","ucv","bcv","SJ")

hist(inp$TAU_FV,breaks=seq(50,400,10),freq=F)
rug(inp$TAU_FV)

for(i in 1:length(bw.method))
{
	d = density(inp$TAU_FV,bw=bw.method[i])
	lines(d$x,d$y,lty=i,col=i)
	print(d)
}

legend("topright",bw.method,col=1:5,lty=1:5)

dev.off()
