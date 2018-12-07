
library(RSQLite)
library(bio3d)

con <- dbConnect(RSQLite::SQLite(),"tmp.gZ1kwmWIre/FV.db")
cond.list <- dbGetQuery(con,"select * from conditions_list")
crd.vels <-  dbGetQuery(con,"select * from crd_vels")
crds.1 <- matrix(data=as.numeric(t(as.matrix(crd.vels[,c("X","Y","Z")]))),
                 nrow=nrow(crd.vels)/max(crd.vels$ATOM_ID),
                 ncol=3*max(crd.vels$ATOM_ID),
                 byrow = T
                )
crds.1 <- as.xyz(crds.1)

con <- dbConnect(RSQLite::SQLite(),"tmp.Jo3VD6LfAC/FV.db")
cond.list <- dbGetQuery(con,"select * from conditions_list")
crd.vels <-  dbGetQuery(con,"select * from crd_vels")
crds.2 <- matrix(data=as.numeric(t(as.matrix(crd.vels[,c("X","Y","Z")]))),
                 nrow=nrow(crd.vels)/max(crd.vels$ATOM_ID),
                 ncol=3*max(crd.vels$ATOM_ID),
                 byrow = T
)
crds.2 <- as.xyz(crds.2)

crds <- rbind(crds.1,crds.2)
crds <- as.xyz(crds)

write.ncdf(10.0*crds,"traj.nc")

pdb <- read.pdb("A.pdb")
phi.sele <- atom.select(pdb,eleno=c(5,7,9,15))
psi.sele <- atom.select(pdb,eleno=c(7,9,15,17))

phi.ang = numeric(length = nrow(crds))
psi.ang = numeric(length = nrow(crds))

for(i in 1:nrow(crds))
{
  phi.ang[i] = torsion.xyz(crds[i,phi.sele$xyz])
  psi.ang[i] = torsion.xyz(crds[i,psi.sele$xyz])
}

library(leaflet)

my.palette <- colorBin(c("gray","black"),bins=50,domain=c(15,460))

pdf("distribution.pdf",paper="a4r",width=13,height=6)

par(mfrow=c(1,2))

plot(phi.ang,psi.ang,asp=1,col=my.palette(cond.list$TAU_FV),pch=20,
     xlim=c(-180,180),ylim=c(-180,180),
     xlab="PHI (degrees)",
     ylab="PSI (degrees)",
     main="ICs following QSD"
     )

kde.dens <- density(cond.list$TAU_FV,bw="SJ")

plot(kde.dens$x,kde.dens$y,type="p",pch=20,col=my.palette(kde.dens$x),
     xlab="Tau F-V (ps)",ylab="Normalised density",
     main="Kernel density estimate with colour coding"
     )
rug(cond.list$TAU_FV)

dev.off()
