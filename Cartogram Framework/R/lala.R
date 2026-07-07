library(sf)
library(cartogram)
library(dplyr)
library(rmapshaper)

# Stage 1: does reading alone work?
world <- st_read("/home/kuenem/Documents/development/lectures/Master Thesis/Cartogram Framework/data/geojson/world/worldTEST.geojson", quiet = TRUE)
print("read OK")

# Stage 2: does the join work?
pop   <- read.csv("/home/kuenem/Documents/development/lectures/Master Thesis/Cartogram Framework/data/statistics/contiguous/world/2010.csv", stringsAsFactors = FALSE)

dworld <- world %>%
  left_join(pop, by = "ISO_CODE") %>%
  rename(pop = `Population..people.`)   # do this immediately after the join

world_proj <- st_transform(dworld, "ESRI:54009")
world_proj <- st_make_valid(world_proj)
world_proj_simplified <- ms_simplify(world_proj, keep = 0.05)

# Now check whether "pop" survived cleanly
colnames(world_proj_simplified)
sum(is.na(world_proj_simplified$pop))
sum(world_proj_simplified$pop <= 0, na.rm = TRUE)

world_cont <- cartogram_cont(world_proj_simplified, weight = "pop", itermax = 20, verbose = TRUE)

st_write(
  world_cont,
  "/home/kuenem/Documents/development/lectures/Master Thesis/Cartogram Framework/R/world_cartogram_2010_cont.geojson",
  driver = "GeoJSON",
  delete_dsn = TRUE
)

world_dorling <- cartogram_dorling(world_proj, weight = "pop")

st_write(
  world_dorling,
  "/home/kuenem/Documents/development/lectures/Master Thesis/Cartogram Framework/R/world_cartogram_2010_dorling.geojson",
  driver = "GeoJSON",
  delete_dsn = TRUE
)

world_ncont <- cartogram_ncont(world_proj, weight = "pop")

st_write(
  world_ncont,
  "/home/kuenem/Documents/development/lectures/Master Thesis/Cartogram Framework/R/world_cartogram_2010_ncont.geojson",
  driver = "GeoJSON",
  delete_dsn = TRUE
)
