install.packages("ggmap")
install.packages("factoextra")

library('dplyr')
library('stats')
library('ggplot2')
library('caret')
library('nnet')
library('ggmap')
library('cluster')
library('fpc')
library('factoextra') 

library(data.table)

rm(list=ls(all=T))
cat('\014')
set.seed(100)

traffic_data <- read.csv(file = "traffic-volume-counts-2012-2013.csv", header = TRUE, sep = ",")

ag_traffic_data <- aggregate(traffic_data, list(Roadway_Name = traffic_data$Roadway.Name,
                                                Id = traffic_data$ID),
                             mean)
ag_traffic_data$Roadway.Name <- NULL
ag_traffic_data$ID <- NULL
capture.output(ag_traffic_data, file = "Aggregated Traffic Data.txt")
sum <- summary(ag_traffic_data[, 3:26])
capture.output(sum, file = "traffic_summary.txt")


scale_traffic_data <- scale(ag_traffic_data[, 3:26])

cluster_traffic_data <- kmeans(scale_traffic_data[,], centers = 3, iter.max = 100, nstart = 25, 
                               algorithm = "Lloyd")
capture.output(str(cluster_traffic_data), file = "traffic clusters.txt")
plotcluster(scale_traffic_data[,], cluster_traffic_data$cluster)

distance <- get_dist(scale_traffic_data)

clusplot(scale_traffic_data[,], cluster_traffic_data$cluster, color = TRUE, shade = TRUE,
        labels = 2, lines = 0)

