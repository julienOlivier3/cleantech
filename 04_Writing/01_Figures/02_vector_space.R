library(tidyverse)
library(ggforce)
library(here)
library(tikzDevice)
source(here("02_Code/util.R"))

start <- c(x = 0, y = 0)

d <- data.frame(
  x = start[c("x", "x", "x")],
  y = start[c("y", "y", "y")],
  xend = c(0.2, 0.125, 0.5), 
  yend = c(1, 0.9, 0.1), 
  type = c("Technology", "Company", "Company"),
  name = c("$\\bar{X_t}$", "$\\bar{Y_1}$", "$\\bar{Y_2}$"))

angles <- with(d, atan2(xend-x, yend-y))

tikz(here("04_Writing/01_Figures/fig_vector_space.tex"),
     height = 16/3,
     width = 16/3)

ggplot() + 
  #geom_point(data = d, aes(x=xend, y=yend), size=4, shape=21, fill="white") +
  geom_text(data = d, aes(x=xend, y=yend, label=name), hjust=0.01, vjust=-0.5) +
  geom_segment(data = d, aes(x=x, y=y, xend=xend, yend=yend, color=type, size=type), arrow=arrow(length = unit(0.1, "inches"))) + 
  geom_arc(aes(x0 = start["x"], y0 = start["y"], r = 0.7, 
               start = angles[1], end = angles[2])) + 
  geom_arc(aes(x0 = start["x"], y0 = start["y"], r = 0.48, 
               start = angles[1], end = angles[3])) + 
  xlim(-0.05,0.6) +
  ylim(-0.05,1.1) +
  scale_color_manual(values = c(goldL, green)) +
  scale_size_manual(values = c(1, 2)) +
  geom_hline(yintercept = 0) +
  geom_vline(xintercept = 0) +
  guides(color=guide_legend(title="Embedding type"), 
         size=guide_legend(title="Embedding type")) +
  annotate("text", x=0.3, y=0.3, label= "$\\theta_2$") + 
  annotate("text", x=0.11, y=0.65, label= "$\\theta_1$") + 
  theme_jod +
  theme(
    legend.position = "top",
    axis.text.x = element_blank(),
    axis.text.y = element_blank(),
    panel.grid.major.y = element_blank(),
    panel.grid.minor.y = element_blank(),
    axis.ticks.x = element_blank(),
    axis.ticks.y = element_blank(),
    axis.title.x = element_blank(),
    axis.title.y = element_blank()) 
  
dev.off()