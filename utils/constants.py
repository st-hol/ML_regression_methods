# Need to specify the headers for this dataset
cols = ["symboling", "normalized_losses", "make", "fuel_type", "aspiration",
        "num_doors", "body_style", "drive_wheels", "engine_location",
        "wheel_base", "length", "width", "height", "curb_weight", "engine_type",
        "num_cylinders", "engine_size", "fuel_system", "bore", "stroke",
        "compression_ratio", "horsepower", "peak_rpm", "city_mpg", "highway_mpg",
        "price"]

# Now lets make things numeric
num_vars = ['normalized_losses', "bore", "stroke", "horsepower", "peak_rpm",
            "price"]

drop_subset = ['price', 'bore', 'stroke', 'horsepower', 'peak_rpm']

z_cols = ['wheel_base', 'length', 'width', 'height',
          'curb_weight', 'engine_size', 'bore', 'stroke', 'horsepower',
          'peak_rpm', 'city_mpg', 'highway_mpg', 'price']
