library(sf)
library(cartogram)
library(dplyr)
library(rmapshaper)

base_dir <- "/home/kuenem/Documents/development/lectures/Master Thesis"
geo_path <- file.path(base_dir, "Cartogram Framework/data/geojson/germany/states.geojson")
stats_path <- file.path(base_dir, "cartogram-cpp/sample_data/germany/states.csv")
out_dir <- file.path(base_dir, "Cartogram Framework/R")

# Edit these values in one place for a new dataset
geojson_key <- "name"              # column in the GeoJSON used for joining
csv_key <- "name"                 # column in the CSV used for joining
weight_column <- "Population"              # numeric column in the CSV used as the cartogram weight
output_name <- "Germany"              # base name for the generated files
target_crs <- 3035                 # WGS 84 / NSIDC EASE-Grid 2.0 Global (equal-area)

geojson <- st_read(geo_path, quiet = TRUE)
print("read OK")

dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)

pop <- read.csv(stats_path, stringsAsFactors = FALSE, check.names = FALSE)

normalize_name <- function(x) {
  x <- trimws(as.character(x))
  x <- gsub("[^[:alnum:]]+", "", x)
  tolower(x)
}

geojson$join_name <- normalize_name(geojson[[geojson_key]])
pop$join_name <- normalize_name(pop[[csv_key]])

# Join on normalized names so spelling and spacing differences do not drop states
geojson_with_data <- geojson %>%
  left_join(pop %>% dplyr::select(join_name, dplyr::all_of(weight_column)), by = "join_name")

# Copy the chosen weight column to a simple name used later in the script
geojson_with_data$pop <- geojson_with_data[[weight_column]]

# Keep only rows with a valid positive weight
geojson_with_data <- geojson_with_data %>%
  filter(!is.na(pop), pop > 0)

# The input data is in lon/lat, which cartogram_cont() cannot use directly.
# Use a projected CRS that matches the region of the data.
geojson_proj <- st_transform(geojson_with_data, target_crs)
geojson_proj <- st_make_valid(geojson_proj)
geojson_proj_simplified <- ms_simplify(
  geojson_proj,
  keep = 0.05,
  keep_shapes = TRUE
)

print(colnames(geojson_proj_simplified))
print(sum(is.na(geojson_proj_simplified$pop)))
print(sum(geojson_proj_simplified$pop <= 0, na.rm = TRUE))

write_output <- function(x, suffix) {
  geojson_path <- file.path(out_dir, paste0(output_name, suffix, ".geojson"))
  svg_path <- file.path(out_dir, paste0(output_name, suffix, ".svg"))
  x_geojson <- st_transform(x, 4326)

  st_write(x_geojson, geojson_path, driver = "GeoJSON", delete_dsn = TRUE)

  grDevices::svg(filename = svg_path, width = 10, height = 7)
  plot(st_geometry(x), col = "#f2f2f2", border = "#666666")
  grDevices::dev.off()
}

time_cartogram <- function(label, expr) {
  start_time <- Sys.time()
  result <- eval.parent(substitute(expr))
  elapsed <- difftime(Sys.time(), start_time, units = "secs")
  cat(sprintf("%s completed in %.2f seconds\n", label, as.numeric(elapsed)))
  result
}

# geojson_cont <- time_cartogram(
#   "input map",
#   cartogram_cont(geojson_proj_simplified, weight = "pop", itermax = 20, verbose = TRUE)
# )
write_output(geojson_proj_simplified, "input")

geojson_cont <- time_cartogram(
  "contiguous cartogram",
  cartogram_cont(geojson_proj_simplified, weight = "pop", itermax = 20, verbose = TRUE)
)
write_output(geojson_cont, "-R")

geojson_dorling <- time_cartogram(
  "Dorling cartogram",
  cartogram_dorling(geojson_proj_simplified, weight = "pop")
)
write_output(geojson_dorling, "-RD")

geojson_ncont <- time_cartogram(
  "non-contiguous cartogram",
  cartogram_ncont(geojson_proj_simplified, weight = "pop")
)
write_output(geojson_ncont, "-RN")
