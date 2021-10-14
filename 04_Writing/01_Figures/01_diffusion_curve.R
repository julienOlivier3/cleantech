library(tidyverse)
library(here)
library(tikzDevice)
source(here("02_Code/util.R"))

# Define some more colors
green <- rgb(0,100,0, alpha = 150, maxColorValue = 255)
grey <- rgb(119,119,119, alpha = 150, maxColorValue = 255)
greyL <- rgb(159,159,159, maxColorValue = 255)
red <- rgb(178,34,34,alpha = 150,  maxColorValue = 255)
redL <- rgb(226,104,104,maxColorValue = 255)
redL2 <- rgb(240,180,180,maxColorValue = 255)
gold <- rgb(255,215,0, alpha = 150, maxColorValue = 255)
goldL <- rgb(254,227,76, maxColorValue = 255)

# Define logistic function 
# Reference: https://www.desmos.com/calculator/agxuc5gip8?lang=de
# higher a: more prolonged curve
# higher k: steeper curve

function_logistic <- function(x, c = 1, a, k){
  y <- c*1/(1+a*exp(-k*x))
  return(y)
}

x <- seq(-1, 1, length.out=1000)

df <- tibble(x = x, 
             `New venture` = function_logistic(x, a = 1, k = 5, c=0.3), 
             `Incumbent` = function_logistic(x, a = 3, k = 5, c=0.2)) %>% 
  pivot_longer(cols = c(`New venture`, `Incumbent`), values_to = "adoption_rate", names_to = "adopter_type") %>% 
  arrange(adopter_type)

tikz(here("04_Writing/01_Figures/fig_diffusion_curve.tex"),
     height = 9/3,
     width = 16/3)

df %>% ggplot() +
  geom_line(aes(x=x, y=adoption_rate, color=adopter_type)) +
  annotate("segment", x=-Inf,xend=Inf,y=0,yend=0,arrow=arrow(length = unit(0.1, "inches"))) +
  annotate("point", x=-1, y=0, cex=5) +
  annotate("text", x=-1, y=0.01, label= "Innovation") + 
  xlab("Time") +
  ylab("Adoption rate") +
  guides(color=guide_legend(title="Adopter type")) +
  scale_color_manual(values = c(redL, goldL)) +
  theme_jod +
  theme(legend.position="top",
        axis.title.x=element_text(size = 9, face = "bold", vjust=10, hjust = 1),
        axis.text.x=element_blank(),
        axis.ticks.x=element_blank(),
        axis.line.y = element_line(colour = 'black', size=0.5, linetype='solid'))
  
dev.off()
  
