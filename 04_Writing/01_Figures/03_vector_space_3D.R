if(!require(pacman)) install.packages(pacman)
p_load(plot3D)


x0 <- c(0, 0, 0, 0)
y0 <- c(0, 0, 0, 0)
z0 <- c(0, 0, 0, 0)
x1 <- c(0.89, -0.46, 0.99, 0.96)
y1 <- c(0.36,  0.88, 0.02, 0.06)
z1 <- c(-0.28, 0.09, 0.05, 0.24)
cols <- c("#1B9E77", "#D95F02", "#7570B3", "#E7298A")

arrows3D(x0, y0, z0, x1, y1, z1, colvar = x1^2, col = cols,
         lwd = 2, d = 3, clab = c("Quality", "score"), 
         main = "Arrows 3D", bty ="g", ticktype = "detailed")
# Add starting point of arrow
points3D(x0, y0, z0, add = TRUE, col="darkred", 
         colkey = FALSE, pch = 19, cex = 1)
# Add labels to the arrows
text3D(x1, y1, z1, c("Sepal.L", "Sepal.W", "Petal.L", "Petal.W"),
       colvar = x1^2, col = cols, add=TRUE, colkey = FALSE)
