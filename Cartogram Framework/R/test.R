library(cartogram)
library(sf)
library(tmap)
library(dplyr)

# ---- 1. Load your own data instead of World ----
world <- st_read("/home/kuenem/Documents/development/lectures/Master Thesis/Cartogram Framework/data/geojson/world/worldTEST.geojson", quiet = TRUE)
pop   <- read.csv("/home/kuenem/Documents/development/lectures/Master Thesis/Cartogram Framework/data/statistics/contiguous/world/2010.csv", stringsAsFactors = FALSE)

# ---- 2. Join population data onto the geometry ----
world <- world %>%
  left_join(pop, by = "ISO_CODE")

# Sanity check: did every polygon get a population value?
sum(is.na(world$`Population (people)`))   # should be 0
# If not 0, inspect which ISO codes failed to match:
world$ISO_CODE[is.na(world$`Population (people)`)]

# ---- 3. Project to an equal-area CRS (required — cartogram functions
#          need projected, not lon/lat, coordinates) ----
# World Mollweide (ESRI:54009) is the standard choice for global cartograms,
# unlike the Africa example's local UTM/Mercator-style choice (EPSG:3395),
# which is fine for one continent but distorts badly at global scale.
world_proj <- st_transform(world, "ESRI:54009")

# ---- 4. Construct the cartograms ----
# Contiguous (Dougenik et al. 1985, rubber-sheet)
# world_cont <- cartogram_cont(world_proj, weight = "Population (people)", itermax = 15)

# Dorling (non-overlapping circles)
world_dorling <- cartogram_dorling(world_proj, weight = "Population (people)")

# Non-contiguous (Olson 1976)
world_ncont <- cartogram_ncont(world_proj, weight = "Population (people)")

# ---- 5. Plot ----
tm_shape(world_cont) +
  tm_polygons("Population (people)", style = "jenks") +
  tm_layout(frame = FALSE, legend.position = c("left", "bottom"))

# ---- 6. Export back to GeoJSON for your Python comparison pipeline ----
st_write(world_cont,    "world_cont.geojson",    delete_dsn = TRUE)
st_write(world_dorling, "world_dorling.geojson", delete_dsn = TRUE)
st_write(world_ncont,   "world_ncont.geojson",   delete_dsn = TRUE)