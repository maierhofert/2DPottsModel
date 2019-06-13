# Thomas Maierhofer

?potts


ncolor <- as.integer(4)
beta <- log(1 + sqrt(ncolor))
beta = 1
theta <- c(rep(0, ncolor), beta)

nrow <- 100
ncol <- 100
x <- matrix(1, nrow = nrow, ncol = ncol)
image(x)

foo <- packPotts(x, ncolor)
out <- potts(foo, theta, boundary = "free", 
             nbatch = 1, blen = 1, nspac = 1, 
             debug = TRUE)
image(unpackPotts(out$final))


str(out)



x <- matrix(sample(4, 2 * 3, replace = TRUE), nrow = 2)
x
foo <- packPotts(x, ncolor = 4)
foo
inspectPotts(foo)
unpackPotts(foo)



########################################
install.packages("bayesImageS")
library("bayesImageS")






















