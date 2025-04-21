# import Pkg
# packages = [
#     "CSV",
#     "DataFrames",
#     "StatsBase",
#     "MultivariateStats",
#     "Flux",
#     "MLDataUtils",
#     "MLUtils",
#     "Statistics",
#     "LinearAlgebra",
#     "PyFormattedStrings",
#     "Random",
#     "ProgressMeter",
#     "Zygote",
#     "Optim",
#     "FluxOptTools",
#     "Plots",
#     "BSON",
#     "CategoricalArrays",
#     "SharedArrays",
#     "SplitApplyCombine",
#     "InvertedIndices",
#     "JSON",
#     "Feather",
#     "Dates",
#     "Metal",
#     "CUDA",
#     "ArgParse",
#     "ModelingToolkit",
#     "TeeStreams"
# ]

# for package in packages
#     Pkg.add(package)
# end

# Import packages
using CSV
using DataFrames
using StatsBase
using MultivariateStats
using Flux
using Flux: params, train!, mse
using Flux.Optimisers
using MLUtils: DataLoader
using MLDataUtils #: splitobs, rescale
using Statistics #: mean, std
using LinearAlgebra #: diag
using PyFormattedStrings
using Base.Threads
using Random
using ProgressMeter
using Zygote, Optim, FluxOptTools
using StatsBase: sample
using Plots
using Plots.PlotMeasures
using BSON: @save, @load
using CategoricalArrays
using SharedArrays
using SplitApplyCombine
using InvertedIndices
using JSON
using Feather
using Dates
using Metal
using CUDA
using CUDA: CuIterator
using ArgParse
using Optim
using ModelingToolkit
using TeeStreams

# Custom AdaGrad optimizer that uses Float32 literals
struct CustomAdaGrad <: Optimisers.AbstractRule
  eta::Float32
  epsilon::Float32
end

# Define the update rule
function Optimisers.apply!(o::CustomAdaGrad, state, x, Δ)
  η, ϵ = o.eta, o.epsilon
  acc = state
  @inbounds @fastmath @. acc += Δ^2
  @inbounds @fastmath @. Δ *= η / (√acc + ϵ)
  return acc
end

Optimisers.init(o::CustomAdaGrad, x::AbstractArray) = fill!(similar(x, Float32), 0f0)

# Define constants
const t_list = [-0.3f0 -0.2f0 -0.1f0 0.3f0 0.6f0 1f0 1.5f0]

function create_folder_with_iterator(path::AbstractString, folder_name::AbstractString; make_new=true)::String
  full_path = joinpath(path, folder_name)
  i = 1
  if isdir(full_path) && !make_new
      return full_path
  end
  while isdir(full_path)
      folder_name_with_iterator = string(folder_name, "_", i)
      full_path = joinpath(path, folder_name_with_iterator)
      i += 1
  end
  mkdir(full_path)
  return full_path
end

function describe(arr)::String
  n = length(arr)
  μ = mean(arr)
  σ = std(arr)
  minimum_value = minimum(arr)
  maximum_value = maximum(arr)
  quartiles = quantile(arr, [0.25f0, 0.5f0, 0.75f0])
  
  return f"n: {n}, mean: {μ:0.6f}, std: {σ:0.6f}, min: {minimum_value:0.6f}, max: {maximum_value:0.6f}, 25%: {quartiles[1]:0.6f}, 50%: {quartiles[2]:0.6f}, 75%: {quartiles[3]:0.6f}"

end

function load_data(infile::String, use_existing_data::Bool, outdir::String, out_streams)::DataFrame
  # Load the data into a DataFrame
  if use_existing_data
    infile = replace(infile, ".feather" => "_balanced.feather")

    println(out_streams, "Using existing data")
  end
  data = Feather.read(infile)
  if !use_existing_data
    println(out_streams, "Loading data...")
    # Remove rows with missing data
    data = data[completecases(data), :]
    println(out_streams, f"Loaded {nrow(data)} rows")
    println(out_streams, f"Data {data[sample(1:nrow(data), 20), :]}")
    for col in names(data)
      if typeof(data[1, col]) == Float64
        println(out_streams, "$col: $(describe(collect(data[:,col])))")
      end
    end

    println(out_streams, "Filtering out extreme values")
    min_vego = minimum(data[!, :v_ego])

    # setup bin ranges
    mm_v_ego = [0.1, 45.0]
    mm_steer_cmd = [-2.0, 2.0]
    mm_lateral_accel = [-4.1, 4.1]
    mm_lateral_jerk = [-5.0, 5.0]
    mm_roll = [-0.20, 0.20]

    # setup bins
    nbins = 121
    # this is the max number of points that will go in each bin. it's low because lowspeed data is scarse.
    sample_size = 20
    step_v_ego = (mm_v_ego[2] - mm_v_ego[1]) / nbins
    step_steer_cmd = (mm_steer_cmd[2] - mm_steer_cmd[1]) / nbins
    step_lateral_accel = (mm_lateral_accel[2] - mm_lateral_accel[1]) / nbins
    step_lateral_jerk = (mm_lateral_jerk[2] - mm_lateral_jerk[1]) / nbins
    step_roll = (mm_roll[2] - mm_roll[1]) / nbins

    function filter_columns(df::DataFrame, partial_match::String, tol)::DataFrame
        for col_name in names(df)
            if occursin(partial_match, col_name)
                # Apply your filter or transformation to the column here
                df = filter(row -> abs(row[col_name]) < tol, df)
            end
        end
        return df
    end

    # if lateral_jerk is always zero, replace with approximation from lateral accel
    if all(data[!, :lateral_jerk] .≈ 0f0)
        println(out_streams, "Replacing lateral_jerk with approximation from lateral_accel")
        data[!, :lateral_jerk] = @fastmath @. (data[!, :lateral_accel_p03] - data[!, :lateral_accel]) / 0.03f0
    end

    # filter data
    old_nrows = nrow(data)
    println(out_streams, f"{old_nrows} rows before filtering")
    data = filter(row -> mm_v_ego[1] < row.v_ego < mm_v_ego[2], data)
    println(out_streams, f"Filtered out {old_nrows - nrow(data)} points with v_ego outside [{mm_v_ego[1]}, {mm_v_ego[2]}]")
    old_nrows = nrow(data)
    data = filter(row -> abs(row.steer_cmd) <= mm_steer_cmd[2], data)
    println(out_streams, f"Filtered out {old_nrows - nrow(data)} points with steer_cmd outside [{-mm_steer_cmd[2]}, {mm_steer_cmd[2]}]")
    old_nrows = nrow(data)
    data = filter_columns(data, "lateral_accel", mm_lateral_accel[2])
    println(out_streams, f"Filtered out {old_nrows - nrow(data)} points with lateral_accel outside [{-mm_lateral_accel[2]}, {mm_lateral_accel[2]}]")
    old_nrows = nrow(data)
    data = filter_columns(data, "lateral_jerk", mm_lateral_jerk[2])
    println(out_streams, f"Filtered out {old_nrows - nrow(data)} points with lateral_jerk outside [{-mm_lateral_jerk[2]}, {mm_lateral_jerk[2]}]")
    old_nrows = nrow(data)
    data = filter_columns(data, "roll", mm_roll[2])
    println(out_streams, f"Filtered out {old_nrows - nrow(data)} points with roll outside [{-mm_roll[2]}, {mm_roll[2]}]")
    println(out_streams, f"{nrow(data)} rows after filtering")

    for col in names(data)
      if typeof(data[1, col]) == Float64
        println(out_streams, "$col: $(describe(collect(data[:,col])))")
      end
    end

    println(out_streams, f"Calculating bins")
    data[!, :v_ego_bins] = cut(data[!, :v_ego], mm_v_ego[1]:step_v_ego:mm_v_ego[2])
    data[!, :lateral_accel_bins] = cut(data[!, :lateral_accel], mm_lateral_accel[1]:step_lateral_accel:mm_lateral_accel[2])
    data[!, :roll_bins] = cut(data[!, :roll], mm_roll[1]:step_roll:mm_roll[2])
    data[!, :lateral_jerk_bins] = cut(data[!, :lateral_jerk], mm_lateral_jerk[1]:step_lateral_jerk:mm_lateral_jerk[2])

    # create a combined column for balancing
    data[!,:combined_column] = string.(data[!,:v_ego_bins], "_", data[!,:lateral_accel_bins])
    
    # balance the data in parallel
    bin_dfs = Vector{Vector{DataFrame}}(undef, Threads.nthreads())
    for i in 1:Threads.nthreads()
        bin_dfs[i] = Vector{DataFrame}()
    end

    unique_bins = unique(data[!, :combined_column])
    prog = ProgressMeter.Progress(length(unique_bins), 1, "Balancing bins:")
    println(out_streams, f"Balancing data into {length(unique_bins)} bins")
    
    # Use threads for better performance
    Threads.@threads for bin_label in unique_bins
        bin_data = data[data[!, :combined_column] .== bin_label, :]
        sampled_bin_data = bin_data[sample(1:nrow(bin_data), min(sample_size, nrow(bin_data))), :]
        push!(bin_dfs[Threads.threadid()], sampled_bin_data)
        
        next!(prog)
    end
    
    flattened_sampled_bin_data = SplitApplyCombine.flatten(bin_dfs)
    data = vcat(flattened_sampled_bin_data...)

    bin_sizes = [size(i,1) for i in flattened_sampled_bin_data]
    println(out_streams, f"Num points after initial balancing: {nrow(data)}")
    bin_min_size = minimum(bin_sizes)
    bin_max_size = maximum(bin_sizes)
    bin_mean_size = mean(bin_sizes)
    bin_std_size = std(bin_sizes)
    println(out_streams, f"Bin sizes: min={bin_min_size}, max={bin_max_size}, mean={bin_mean_size}, std={bin_std_size}")

    new_bin_size = Int(round(max((bin_min_size + bin_mean_size)/2, mean(bin_sizes) - bin_std_size/2), digits=0))
    println(out_streams, f"Shrinking bins to {new_bin_size} points")
    
    # Pre-allocate for better performance
    resized_bin_data = Vector{DataFrame}(undef, length(flattened_sampled_bin_data))
    
    Threads.@threads for i in 1:length(flattened_sampled_bin_data)
        resized_bin_data[i] = flattened_sampled_bin_data[i][sample(1:size(flattened_sampled_bin_data[i],1), new_bin_size), :]
    end
    
    data = vcat(resized_bin_data...)
    println(out_streams, f"Num points after final balancing: {nrow(data)}")

    for col in names(data)
      if typeof(data[1, col]) == Float64
        println(out_streams, "$col: $(describe(collect(data[:,col])))")
      end
    end

    Feather.write(joinpath(outdir, replace(infile, ".feather" => "_balanced.feather")), data)
  else
    println(out_streams, "Loading preprocessed data...")
  end

  return data
end

function train_model(working_dir::String, use_existing_model::Bool, data::DataFrame, out_streams)::NamedTuple{(:model, :input_mean, :input_std, :X_train, :y_train, :X_test, :y_test, :test_loss), Tuple{Flux.Chain, Matrix{Float32}, Matrix{Float32}, Matrix{Float32}, Vector{Float32}, Matrix{Float32}, Vector{Float32}, Float32}}
  model_path = joinpath(working_dir, Base.basename(working_dir))

  # split into train and test sets
  train, test = stratifiedobs(row->row[:combined_column], data, p = 0.8)

  # remove columns used only for binning
  select!(data, Not([:combined_column, :v_ego_bins, :lateral_accel_bins, :lateral_jerk_bins, :roll_bins]))

  # split into independent and dependent variables
  println(out_streams, select(data, Not([:steer_cmd]))[sample(1:nrow(data), 10), :])
  X = Float32.(Matrix(select(data, Not([:steer_cmd]))))
  y = Float32.(data[:, :steer_cmd];)

  # Calculate the mean and standard deviation of the input features
  input_mean = mean(X, dims=1)
  input_std = std(X, dims=1)

  # Create a copy of the DataFrames with the signs of steer_cmd, lateral_accel, lateral_jerk, and roll reversed to make the data symmetric
  old_size = size(train, 1)
  println(out_streams, "Training data before copying symmmetric data: $old_size")
  data_sym = deepcopy(train)
  # symmetrize data by flipping the sign of all columns except v_ego and a_ego in one line
  @views data_sym[!, Not([:v_ego])] = -1f0 .* data_sym[!, Not([:v_ego])]
  train = vcat(train, data_sym)
  println(out_streams, "Training data after copying symmmetric data: $(size(train,1))")

  old_size = size(test, 1)
  println(out_streams, "Test data before copying symmmetric data: $old_size")
  data_sym = deepcopy(test)
  @views data_sym[!, Not([:v_ego])] = -1f0 .* data_sym[!, Not([:v_ego])]
  test = vcat(test, data_sym)
  println(out_streams, "Test data after copying symmmetric data: $(size(test,1))")

  # Check for Metal first (Apple Silicon), then CUDA, then fall back to CPU
  if Metal.functional()
    device = gpu
    Metal.allowscalar(false)
    println(out_streams, "Using device: Metal GPU")
  elseif CUDA.functional()
    device = gpu
    CUDA.allowscalar(false)
    println(out_streams, "Using device: CUDA GPU")
  else
    device = cpu
    println(out_streams, "Using device: CPU")
  end

  # normalize the data - use @views to avoid unnecessary copies and convert directly to Float32
  X_train = Matrix{Float32}(select(train, Not([:steer_cmd])))
  @fastmath @. X_train = (X_train - input_mean) / input_std
  X_train = X_train |> device
  
  y_train = Array{Float32}(train[:, "steer_cmd"]) |> device
  
  X_test = Matrix{Float32}(select(test, Not([:steer_cmd])))
  @fastmath @. X_test = (X_test - input_mean) / input_std
  X_test = X_test |> device
  
  y_test = Array{Float32}(test[:, "steer_cmd"]) |> device

  input_dim = size(X_train, 2)

  # Define the model
  model = Chain(
      Dense(input_dim, 7, sigmoid),
      Dense(7, 13, sigmoid),
      Dense(13, 3),
      Dense(3, 1)
  ) |> device

  # Define the loss function, which includes penalties to enforce physically correct behavior

  # Define the range of values for each independent variable
  speed_len = 8
  other_len = 9
  v_ego_range = range(1f0, stop=40f0, length=speed_len)
  lateral_acceleration_range = range(-4f0, stop=4f0, length=other_len)
  lateral_acceleration_range_hi = range(-3.95f0, stop=4.05f0, length=other_len)
  lateral_jerk_range = range(-5f0, stop=5f0, length=other_len)
  lateral_jerk_range_hi = range(-4.87f0, stop=5.13f0, length=other_len)
  lateral_error_range = range(-5f0, stop=5f0, length=other_len)
  lateral_error_range_hi = range(-4.87f0, stop=5.13f0, length=other_len)
  roll_range = range(-0.2f0, stop=0.2f0, length=other_len)
  roll_range_hi = range(-0.17f0, stop=0.23f0, length=other_len)
  roll_rate_range = range(-0.4f0, stop=0.4f0, length=other_len)
  roll_rate_range_hi = range(-0.35f0, stop=0.45f0, length=other_len)

  function prepare_test_grid(lat_jerk_func, v_ego_range, lateral_acceleration_range, lateral_jerk_range, roll_range, lateral_error_range, roll_rate_range)
    num_test_samples = size(v_ego_range, 1) * size(lateral_acceleration_range, 1) * size(lateral_jerk_range, 1) * size(roll_range, 1) * size(lateral_error_range, 1) * size(roll_rate_range, 1)
    out_grid = Matrix{Float32}(undef, 4 + 2 * size(t_list,2), num_test_samples)
    i = 1
    
    # Pre-allocate arrays for better performance
    lat_accels = Vector{Float32}(undef, length(t_list))
    rolls = Vector{Float32}(undef, length(t_list))
    grid_tmp = Vector{Float32}(undef, 4 + 2 * length(t_list))
    
    for la in lateral_acceleration_range
      for lj in lateral_jerk_range
        for le in lateral_error_range
          for roll in roll_range
            for roll_rate in roll_rate_range
              for v_ego in v_ego_range
                # Use in-place operations to avoid allocations
                @inbounds for (idx, t) in enumerate(t_list)
                  lat_accels[idx] = lat_jerk_func(la, lj, t)
                  rolls[idx] = roll + roll_rate * t
                end
                
                # Construct grid_tmp without intermediate allocations
                grid_tmp[1] = v_ego
                grid_tmp[2] = la
                grid_tmp[3] = le + lj
                grid_tmp[4] = roll
                
                @inbounds for idx in 1:length(t_list)
                  grid_tmp[4 + idx] = lat_accels[idx]
                  grid_tmp[4 + length(t_list) + idx] = rolls[idx]
                end
                
                # Normalize and store in out_grid
                @inbounds @fastmath for j in 1:length(grid_tmp)
                  out_grid[j, i] = (grid_tmp[j] - input_mean[j]) / input_std[j]
                end
                
                i += 1
              end
            end
          end
        end
      end
    end
    return out_grid
  end

  # Define the linear jerk function for better performance - use oftype for type stability
  lj_func = (la, lj, t) -> la + oftype(la, lj * t)
  
  # Prepare all the grids
  grid = prepare_test_grid(lj_func, v_ego_range, lateral_acceleration_range, lateral_jerk_range, roll_range, lateral_error_range, roll_rate_range)
  grid_da = prepare_test_grid(lj_func, v_ego_range, lateral_acceleration_range_hi, lateral_jerk_range, roll_range, lateral_error_range, roll_rate_range)
  grid_dj = prepare_test_grid(lj_func, v_ego_range, lateral_acceleration_range, lateral_jerk_range_hi, roll_range, lateral_error_range, roll_rate_range)
  grid_de = prepare_test_grid(lj_func, v_ego_range, lateral_acceleration_range, lateral_jerk_range, roll_range, lateral_error_range_hi, roll_rate_range)
  grid_dg = prepare_test_grid(lj_func, v_ego_range, lateral_acceleration_range, lateral_jerk_range, roll_range_hi, lateral_error_range, roll_rate_range)
  grid_dg_rate = prepare_test_grid(lj_func, v_ego_range, lateral_acceleration_range, lateral_jerk_range, roll_range_hi, lateral_error_range, roll_rate_range_hi)
  grid_odd_neg = prepare_test_grid(lj_func, v_ego_range, -lateral_acceleration_range, -lateral_jerk_range, -roll_range, -lateral_error_range, -roll_rate_range)
  grid_origin = prepare_test_grid(lj_func, v_ego_range, 0, 0, 0, 0, 0)

  varnames = join(names(select(data, Not([:steer_cmd]))), ", ")

  function physical_constraint_losses(x, y_pred, λ_monotonic::Float32, λ_odd::Float32, λ_origin::Float32) :: Float32
      if λ_monotonic ≈ 0f0 && λ_odd ≈ 0f0 && λ_origin ≈ 0f0
          return 0f0
      end
      
      monotonicity_loss = 0f0
      odd_loss = 0f0
      origin_loss = 0f0

      model_grid = model(grid)
      
      if λ_monotonic ≉ 0f0
          model_da = model(grid_da)
          model_dj = model(grid_dj)
          model_de = model(grid_de)
          model_dg = model(grid_dg)
          model_dgr = model(grid_dg_rate)

          # d(output)/d(lat accel/jerk) and d(output)/d(error) should be positive,
          # d(output)/d(roll) and d(output)/d(roll_rate) should be negative
          @fastmath monotonicity_loss = sum(abs2, max.(0f0, (model_da .- model_grid) .* -1f0)) +
                              sum(abs2, max.(0f0, (model_dj .- model_grid) .* -1f0)) +
                              sum(abs2, max.(0f0, (model_de .- model_grid) .* -1f0)) +
                              sum(abs2, max.(0f0, model_dg .- model_grid)) +
                              sum(abs2, max.(0f0, model_dgr .- model_grid))  
      end
      
      if λ_odd ≉ 0f0
          model_odd_neg = model(grid_odd_neg)
          @fastmath odd_loss = sum(abs2, model_grid .+ model_odd_neg)
      end

      if λ_origin ≉ 0f0
          @fastmath origin_loss = sum(abs2, model(grid_origin))
      end

      @fastmath return λ_monotonic * monotonicity_loss + λ_odd * odd_loss + λ_origin * origin_loss
  end

  function model_with_params(model, params)
      temp_model = deepcopy(model)
      Flux.loadparams!(temp_model, params)
      return temp_model
  end

  function get_scaling_vector(x, low, high)
      speed = @view x[:, 1]
      min_speed, max_speed = extrema(speed)
      @fastmath normalized_speed = @. (speed - min_speed) / (max_speed - min_speed) # normalize to [0, 1]
      @fastmath scaling_vector = @. low + (high - low) * normalized_speed # scale to [low, high]
      return scaling_vector
  end

  function custom_mse(y_true, y_pred, scaling_vector)
      @fastmath residuals = y_true .- y_pred
      @fastmath squared_errors = residuals .* residuals
      @fastmath weighted_squared_errors = scaling_vector .* squared_errors
      return sum(weighted_squared_errors) / oftype(weighted_squared_errors[1], length(y_true))
  end

  function combined_loss(x, y_true, y_pred, model, λ::Float32, λ_monotonic::Float32, λ_odd::Float32, λ_origin::Float32, low::Float32, high::Float32)
      mse = 0f0
      if low ≉ high
        scaling_vector = get_scaling_vector(x, low, high)
        mse = custom_mse(y_true, y_pred, scaling_vector)
      else
        mse = Flux.Losses.mse(y_true, y_pred)
      end
      
      l2 = λ ≈ 0f0 ? 0f0 : λ * sum(p -> sum(abs2, p), params(model))
      physical_constraints = physical_constraint_losses(x, y_pred, λ_monotonic, λ_odd, λ_origin)
      
      @fastmath return mse + l2 + physical_constraints
  end

  loss(x, y, model, λ::Float32=0f0, λ_monotonic::Float32=0f0, λ_odd::Float32=0f0, λ_origin::Float32=0f0, low::Float32=1f0, high::Float32=1f0) = combined_loss(x, y', model(x), model, λ, λ_monotonic, λ_odd, λ_origin, low, high)

  # pick an optimizer
  # opt = Flux.ADAM(0.001)
  # opt = Flux.Nesterov()
  opt = Flux.AdaGrad()
  state_tree = Optimisers.setup(opt, model)

  # train the model (with batches of shuffled data)
  tol = log10(size(X_train, 1)) > 7 ? 1f-4 : 1f-6
  Δtol = log10(size(X_train, 1)) > 7 ? 1f-4 : 5f-5
  logstep = device == gpu ? 50 : log10(size(X_train, 1)) > 7 ? 3 : 10
  logstepgrowth = 1
  logstepfloat = Float32(logstep)
  Δloss = Inf32
  ΔΔloss = Inf32
  Δloss_last = 0f0
  loss_last = Inf32
  loss_cur = 0f0
  stall_check_count = device == gpu ? 50000 : 15
  stall_count = 0
  epoch = 1
  ilog = logstep + 1
  epoch_max = device == gpu ? 10000 : log10(size(X_train, 1)) > 6 ? 150 : 1000
  epoch_min = 25
  batch_size = 200000

  println(out_streams, size(X_train))
  println(out_streams, size(y_train))

  if use_existing_model
      old_model = "$model_path.bson"
      println(out_streams, "Loading old model, $old_model")
      @load old_model model
  else

    for col in names(data)
      if typeof(data[1, col]) == Float64
        println(out_streams, "$col: $(describe(collect(data[:,col])))")
      end
    end

    if size(y_train, 1) > batch_size
      X_train = cpu(X_train)
      y_train = cpu(y_train)
    end
    train_data_loader = DataLoader((X_train', y_train), batchsize=batch_size, shuffle=true)

    grid = grid |> device
    grid_da = grid_da |> device
    grid_dj = grid_dj |> device
    grid_de = grid_de |> device
    grid_dg = grid_dg |> device
    grid_dg_rate = grid_dg_rate |> device
    grid_odd_neg = grid_odd_neg |> device
    grid_origin = grid_origin |> device
    
    λmax = 0.000f0
    λ_monotonicmax = 0.0015f0
    λ_oddmax = 0.000001f0
    λ_originmax = 0.0001f0

    λ_start_epoch_fraction = 0.25f0
    λ_monotonic_start_epoch_fraction = 0.6f0
    λ_odd_start_epoch_fraction = 0.5f0
    λ_origin_start_epoch_fraction = 0.7f0

    start_time = now()
    last_log_time = start_time - Dates.Millisecond(40000)
    ptime(t) = Dates.format(t, "HH:MM:SS")
    losses = Float32[]
    lambdas = NTuple{4, Float32}[]
    ilog = 0
    epoch_last = 1

    while epoch < epoch_min || (epoch < epoch_max) # && (abs(Δloss) > tol || abs(ΔΔloss) > Δtol)
        # determine λ_monotonic and λ_odd. They stay at 0 until 25% of the way through the training, then increase linearly to their max values by 75% of the way through the training
        λ = λmax * min(1f0, max(0f0, epoch - epoch_max * λ_start_epoch_fraction) / (epoch_max * 0.7f0)) |> device
        λ_monotonic = λ_monotonicmax * min(1f0, max(0f0, epoch - epoch_max * λ_monotonic_start_epoch_fraction) / (epoch_max * 0.3f0)) |> device
        λ_odd = λ_oddmax * min(1f0, max(0f0, epoch - epoch_max * λ_odd_start_epoch_fraction) / (epoch_max * 0.4f0)) |> device
        λ_origin = λ_originmax * min(1f0, max(0f0, epoch - epoch_max * λ_origin_start_epoch_fraction) / (epoch_max * 0.2f0)) |> device
        l = 0f0
        if size(y_train, 1) > batch_size
          # Handle batch processing for both Metal and CUDA
          for (x, y) in train_data_loader
            for ibatch in 1:20
              gs = Flux.gradient(model, X_train, y_train, λ, λ_monotonic, λ_odd, λ_origin) do model, X_train, y_train, λ, λ_monotonic, λ_odd, λ_origin
                l = loss(x, y, model, λ, λ_monotonic, λ_odd, λ_origin)
              end
              state_tree, model = Optimisers.update!(state_tree, model, gs[1])
              if ibatch % 2 == 0
                epoch += 1
                ilog += 1
                push!(losses, l)
                push!(lambdas, (λ, λ_monotonic, λ_odd, λ_origin))
                if epoch > epoch_max
                  break
                end
              end
            end
            if epoch > epoch_max
              break
            end
          end
        else
          for (x, y) in train_data_loader
            gs = Flux.gradient(model, λ, λ_monotonic, λ_odd, λ_origin) do model, λ, λ_monotonic, λ_odd, λ_origin
              l = loss(x, y, model, λ, λ_monotonic, λ_odd, λ_origin)
            end;
            state_tree, model = Optimisers.update!(state_tree, model, gs[1])
            GC.gc()
          end
        end

        push!(losses, l)
        push!(lambdas, (λ, λ_monotonic, λ_odd, λ_origin))
      
        t = now()
        if (t - last_log_time) > Dates.Millisecond(10000) || epoch >= epoch_max
            loss_cur = l
            Δloss = loss_cur - loss_last
            if (Δloss > 0) || (Δloss ≈ 0)
                stall_count += 1
            end
            if stall_count ≥ stall_check_count && ((Δloss < 0) || (Δloss ≈ 0f0))
                println(out_streams, "Stalled at epoch $epoch, loss $loss_cur")
                break
            end
            ΔΔloss = Δloss - Δloss_last
            loss_last = loss_cur
            Δloss_last = Δloss
            if abs(Δloss) > tol || abs(Δloss_last) > Δtol
                cur_time = Dates.format(now(), "HH:MM:SS")
                # predict time remaining
                time_str = "estimating remaining time..."
                if epoch / epoch_max > 0.05
                    elapsed = (t - last_log_time).value / 1000 # Milliseconds to seconds
                    epochs = epoch - epoch_last
                    epoch_remaining = epoch_max - epoch
                    recent_rate = epochs / elapsed
                    epoch_remaining_time = Dates.Millisecond(1000 * round(epoch_remaining / recent_rate))
                    total_time = epoch_remaining_time + Dates.Millisecond(1000 * round((t - start_time).value / 1000))
                    epoch_remaining_time_str = Dates.canonicalize(Dates.CompoundPeriod(epoch_remaining_time))
                    epoch_total_time_str = Dates.canonicalize(Dates.CompoundPeriod(total_time))
                    time_str = "$epoch_remaining_time_str remaining of $epoch_total_time_str total"
                end
                println(out_streams, f"{cur_time} Epoch {epoch:3d} of {epoch_max} ({time_str}); Loss: {loss_cur:.6f}, ΔLoss: {Δloss:.7f}, ΔΔLoss: {ΔΔloss:.9f}, λ: {λ:.6G}, λ_monotonic: {λ_monotonic:.6G}, λ_odd: {λ_odd:.6G}, λ_origin: {λ_origin:.6G}")
            end
            logstepfloat *= logstepgrowth
            logstep = round(Int, logstepfloat)
            last_log_time = t
            epoch_last = epoch
        end
        epoch += 1
        ilog += 1
    end
    cur_time = Dates.format(now(), "HH:MM:SS")
    println(out_streams, f"round 1 {cur_time} Epoch: {epoch:3d} (of {epoch_max}; Loss: {loss_cur:.6f}, ΔLoss: {Δloss:.7f}, ΔΔLoss: {ΔΔloss:.9f}")

    # save the model for easy loading back into Flux.jl
      # bring data back to cpu
    if device == gpu
      X_train = cpu(X_train)
      y_train = cpu(y_train)
      X_test = cpu(X_test)
      y_test = cpu(y_test)
      model = cpu(model)
      grid = cpu(grid)
      grid_da = cpu(grid_da)
      grid_dj = cpu(grid_dj)
      grid_de = cpu(grid_de)
      grid_dg = cpu(grid_dg)
      grid_dg_rate = cpu(grid_dg_rate)
      grid_odd_neg = cpu(grid_odd_neg)
      grid_origin = cpu(grid_origin)
    end

    @save "$model_path.bson" model

    # Create and save plot of training

    # x value is just the epoch number
    x = 1:length(losses)

    mean_loss = mean(losses)
    std_loss = std(losses)

    # Plot the loss values with a black line on the left axis
    p = plot(x, losses, label="Loss", color=:black, ylabel="Loss", xlabel="Epoch", yscale=:log10, legend=:topleft)

    # Iterate through each lamda and plot it with different colors
    colors = [:red, :blue, :green, :orange]
    loss_names = ["λ_l2_regularization", "λ_monotonic", "λ_odd", "λ_origin"]
    for i in 1:4
        lambda = [l[i] for l in lambdas]
        l_max = maximum(lambda)
        if l_max == 0f0
            continue
        end
        normalized_lambda = lambda ./ l_max
        plot!(twinx(), normalized_lambda, label=loss_names[i], color=colors[i], ylabel="Normalized Lamda",xticks=:none, legend=:bottomright)
    end
    savefig(p, "$model_path.training.png")
    
    println(out_streams, "Finished after $epoch epochs, Loss: $loss_cur, ΔLoss: $Δloss, Test loss: $(loss(X_test', y_test, model))")

  end

  function feedforward_function(input_data; zero_bias=false)
    # Scale the input data using the stored mean and standard deviation values
    input_data_scaled = (input_data .- input_mean) ./ input_std
    if zero_bias
      eval_model = deepcopy(model)
      for layer in eval_model.layers
        if layer isa Dense
          Flux.params(layer)[2] .= zeros(Float32, size(Flux.params(layer)[2]))
        end
      end
      steer_command = eval_model(input_data_scaled')
      return steer_command[1]
    else
      steer_command = model(input_data_scaled')
      return steer_command[1]
    end
  end

  function evaluate_manually(model, x; zero_bias=false)
    for layer in model.layers
      W, b = params(layer)
      W = W'
      b = b'
      if zero_bias
        b = zeros(Float32, size(b))
      end
      if layer.σ == σ
        x = σ.(x * W .+ b)
      elseif layer.σ == identity
        x = identity.(x * W .+ b)
      elseif layer.σ == tanh
        x = tanh.(x * W .+ b)
      elseif layer.σ == leakyrelu
        x = leakyrelu.(x * W .+ b)
      else
        try
          x = layer.σ.(x * W .+ b)
        catch e
          println(out_streams, "Unsupported activation function: $(layer.σ)")
          rethrow(e)
        end
      end
    end
    return x[1]
  end

  function feedforward_function_manual(input_data; zero_bias=false)
    # Scale the input data using the stored mean and standard deviation values
    input_data_scaled = (input_data .- input_mean) ./ input_std
    steer_command = evaluate_manually(model, input_data_scaled, zero_bias=zero_bias)
    return steer_command
  end

  function test_evaluate_manually(model; zero_bias=false)
    vego_range = 0f0:20f0:40f0
    lataccel_range = -4f0:4f0:4f0
    latjerk_range = -4f0:4f0:4f0
    roll_range = -0.2f0:0.2f0:0.2f0
    println(out_streams, "Testing manual model evaluation (as performed in OpenPilot)...")
    println(out_streams, "Testing with zero bias: $zero_bias")
    test_dict = Dict()
    for vego in vego_range
      for lataccel in lataccel_range
        for latjerk in latjerk_range
          for roll in roll_range
            lat_accels = [lataccel + t * latjerk for t in t_list]
            rolls = [roll for t in t_list]
            input_data = [vego lataccel latjerk roll]
            x = hcat(input_data, lat_accels, rolls)
            xstr = "[" * join(x, ",") * "]"
            result_model = feedforward_function(x, zero_bias=zero_bias)  # Model evaluation
            result_manual = feedforward_function_manual(x, zero_bias=zero_bias)  # Manual evaluation
            test_dict[xstr] = result_model
            if result_model ≉ result_manual
              println(out_streams, "Mismatch at input: $x")
              println(out_streams, "Model: $result_model, Manual: $result_manual")
              return false
            end
          end
        end
      end
    end

    println(out_streams, "Test passed: All outputs match!")
    return test_dict
  end

  test_dict_zero_bias = test_evaluate_manually(model, zero_bias=true)
  test_dict = test_evaluate_manually(model)

  current_date_and_time = Dates.format(now(), "yyyy-mm-dd_HH-MM-SS")

  model_test_loss = loss(X_test', y_test, model)

  # save model to json for Python import
  function export_model_params_to_json(model::Chain, input_mean::Matrix{Float32}, input_std::Matrix{Float32}, filename::String, current_date_and_time, model_test_loss, input_vars)
      W, b = params(model.layers[1])
      input_size = size(W, 2)
      output_size = size(params(model.layers[end])[1], 1)
      params_dict = Dict{String, Any}("input_size" => input_size, "output_size" => output_size, "layers" => [], "input_mean" => input_mean, "input_std" => input_std, "current_date_and_time" => current_date_and_time, "model_test_loss" => model_test_loss, "input_vars" => input_vars)

      for (idx, layer) in enumerate(model.layers)
          if isa(layer, Dense)
              W, b = params(layer)
              params_dict["layers"] = push!(params_dict["layers"], Dict(
                  "dense_$(idx)_W" => Array{Float32}(W'),
                  "dense_$(idx)_b" => Array{Float32}(b'),
                  "activation" => string(layer.σ)
              ))
          end
      end

      open(filename, "w") do f
          write(f, JSON.json(params_dict))
      end
  end

  export_model_params_to_json(model, Matrix{Float32}(input_mean), Matrix{Float32}(input_std), "$model_path.json", current_date_and_time, model_test_loss, names(select(data, Not([:steer_cmd]))))


  # Evaluate the model on the test set 
  test_loss = loss(X_test', y_test, model)
  println(out_streams, "Test loss (MSE): ", test_loss)

  return (model=model, input_mean=input_mean, input_std=input_std, X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test, test_loss=test_loss)

end

function test_plot_model(model::Flux.Chain, plot_path::String, X_train::Matrix{Float32}, y_train::Vector{Float32}, X_test::Matrix{Float32}, y_test::Vector{Float32}, input_mean::Matrix{Float32}, input_std::Matrix{Float32}, x_var_names::String, out_streams, test_loss::Float32)

  car_name = Base.basename(plot_path)

  function feedforward_function(input_data)
    # Scale the input data using the stored mean and standard deviation values
    @fastmath input_data_scaled = @. (input_data - input_mean) / input_std
    steer_command = model(input_data_scaled')
    return steer_command[1]
  end

  max_abs_lat_jerk = 0.2
  max_abs_roll = 0.03

  # Create a function to filter the dataset based on speed
  function filter_data_by_speed(Xi, yi, speed, tolerance; no_jerk=false, no_roll=false, shuffle_data=true)
    indices = findall(@. abs(Xi[:, 1] - speed) < tolerance)
    X, y = Xi[indices, :], yi[indices]
    if no_jerk
      indices = findall(@. abs(X[:, 3]) < max_abs_lat_jerk)
      X, y = X[indices, :], y[indices]
    end
    if no_roll
      indices = findall(@. abs(X[:, 4]) < max_abs_roll)
      X, y = X[indices, :], y[indices]
    end
    if shuffle_data
      indices = shuffle(1:size(X, 1))
      X, y = X[indices, :], y[indices]
    end
    return X, y
  end

  @fastmath X_train_rescaled = @. X_train * input_std + input_mean
  @fastmath X_test_rescaled = @. X_test * input_std + input_mean

  scatter_points_desired = 1000

  # Iterate over the speed range and create a plot for each speed
  # first w.r.t. lateral jerk
  speed_step = 6
  speed_range = 3:speed_step:35
  lateral_acceleration_range = range(-4.0, 4.0, length=100)

  marker_alpha = 0.1
  test_alpha = 0.25
  train_color = :black
  test_color = :cadetblue
  cpalette = :Dark2_5
  line_width = 3

  plot_col_num = 1
  p = plot(layout = (size(collect(speed_range), 1), 3), legend=:bottomright, size=(2300, 2300), margin=8mm)

  # Pre-allocate arrays for better performance
  lat_accels = Vector{Float32}(undef, length(t_list))
  rolls = Vector{Float32}(undef, length(t_list))

  # Iterate over the speed range and create a plot for each speed
  for (si, speed) in enumerate(speed_range)
    # Plot the training data
    X_train_filtered, y_train_filtered = filter_data_by_speed(X_train_rescaled, y_train, speed, speed_step/2, no_roll=true)
    plot_scatter_step = round(Int, max(1, size(X_train_filtered, 1) / scatter_points_desired / 2))
    scatter!(p[si,plot_col_num], X_train_filtered[1:plot_scatter_step:end, 2], y_train_filtered[1:plot_scatter_step:end], label="Training Data", markersize=2, markercolor=train_color, markeralpha=marker_alpha, xlims=(-3.5, 3.5), ylims=(-1.4,1.4), markerstrokewidths=0)

    # Plot the test data
    X_test_filtered, y_test_filtered = filter_data_by_speed(X_test_rescaled, y_test, speed, speed_step/2, no_roll=true)
    plot_scatter_step = round(Int, max(1, size(X_test_filtered, 1) / scatter_points_desired))
    scatter!(p[si,plot_col_num], X_test_filtered[1:plot_scatter_step:end, 2], y_test_filtered[1:plot_scatter_step:end], label="Test Data", markersize=2, markercolor=test_color, markeralpha=test_alpha, xlims=(-3.5, 3.5), ylims=(-1.4,1.4), markerstrokewidths=0)

    vline!(p[si,plot_col_num], [0.0], color=:black, linewidth=1, label="")
    hline!(p[si,plot_col_num], [0.0], color=:black, linewidth=1, label="")
    hline!(p[si,plot_col_num], [-1, 1], color=:red, linewidth=1, label="")
    
    # Plot "sustained error", so that the amount of lat accel error propogates backwards and forwards through time
    # (i.e. you're entering/exiting a turn at a constant rate that the car isn't keeping up with).
    # Here, we set a lateral jerk value and use that to compute the lateral acceleration at each time step.
    ci = 1
    for lj in [-1.0, -0.25, 0.0, 0.25, 1.0]
      x_model = Vector{Float32}(undef, length(lateral_acceleration_range))
      y_model = Vector{Float32}(undef, length(lateral_acceleration_range))
      
      for (i, la) in enumerate(lateral_acceleration_range)
          # Fill pre-allocated arrays
          fill!(lat_accels, la)
          fill!(rolls, 0f0)
          
          # Create input data
          input_data = [speed la lj 0.0]
          input_data = hcat(input_data, lat_accels', rolls')
          
          # Get model prediction
          steer_command = feedforward_function(input_data)
          x_model[i] = la
          y_model[i] = steer_command
      end
      
      plot!(p[si,plot_col_num], x_model, y_model, label="err = $lj", linewidth=line_width, xlims=(-3.5, 3.5), ylims=(-1.4,1.4), color=palette(cpalette,5)[ci])
      ci += 1
    end

    # Configure the plot's appearance
    if si == 1
      title!(p[1,plot_col_num], f"{car_name}\nLateral acceleration/jerk error response\n{(speed-speed_step/2)*2.24:.2G}-{(speed+speed_step/2)*2.24:.2G} mph w/ |roll| < {max_abs_roll:.2G}")
    else
      title!(p[si,plot_col_num], f"{(speed-speed_step/2)*2.24:.2G}-{(speed+speed_step/2)*2.24:.2G} mph w/ |roll| < {max_abs_roll:.2G}")
    end
    xlabel!(p[si,plot_col_num], "lateral acceleration (m/s²; [+] = right turn)")
    ylabel!(p[si,plot_col_num], "steer command\n([+] = pushing right)")
  end

  plot_col_num += 1

  for (si, speed) in enumerate(speed_range)

    # Plot the training data
    X_train_filtered, y_train_filtered = filter_data_by_speed(X_train_rescaled, y_train, speed, speed_step/2, no_roll=true)
    plot_scatter_step = round(Int, max(1, size(X_train_filtered, 1) / scatter_points_desired / 2))
    scatter!(p[si,plot_col_num], X_train_filtered[1:plot_scatter_step:end, 2], y_train_filtered[1:plot_scatter_step:end], label="Training Data", markersize=2, markercolor=train_color, markeralpha=marker_alpha, xlims=(-3.5, 3.5), ylims=(-1.4,1.4), markerstrokewidths=0)

    # Plot the test data
    X_test_filtered, y_test_filtered = filter_data_by_speed(X_test_rescaled, y_test, speed, speed_step/2, no_roll=true)
    plot_scatter_step = round(Int, max(1, size(X_test_filtered, 1) / scatter_points_desired))
    scatter!(p[si,plot_col_num], X_test_filtered[1:plot_scatter_step:end, 2], y_test_filtered[1:plot_scatter_step:end], label="Test Data", markersize=2, markercolor=test_color, markeralpha=test_alpha, xlims=(-3.5, 3.5), ylims=(-1.4,1.4), markerstrokewidths=0)

    vline!(p[si,plot_col_num], [0.0], color=:black, linewidth=1, label="")
    hline!(p[si,plot_col_num], [0.0], color=:black, linewidth=1, label="")
    hline!(p[si,plot_col_num], [-1, 1], color=:red, linewidth=1, label="")
    
    # Plot "sustained error", so that the amount of lat accel error propogates backwards and forwards through time
    ci = 1
    for lj in -1.0:0.5:1.0
      x_model = Vector{Float32}(undef, length(lateral_acceleration_range))
      y_model = Vector{Float32}(undef, length(lateral_acceleration_range))
      
      for (i, la) in enumerate(lateral_acceleration_range)
          # Calculate lat_accels with lateral jerk
          @inbounds for (idx, t) in enumerate(t_list)
              lat_accels[idx] = la + lj * t
              rolls[idx] = 0f0
          end
          
          input_data = [speed la 0.0 0.0]
          input_data = hcat(input_data, lat_accels', rolls')
          steer_command = feedforward_function(input_data)
          
          x_model[i] = la
          y_model[i] = steer_command
      end
      
      plot!(p[si,plot_col_num], x_model, y_model, label="lat. jerk = $lj", linewidth=line_width, xlims=(-3.5, 3.5), ylims=(-1.4,1.4), color=palette(cpalette,5)[ci])
      ci += 1
    end

    # Configure the plot's appearance
    if si == 1
      title!(p[1,plot_col_num], f"{x_var_names}\nLateral jerk response\n{(speed-speed_step/2)*2.24:.2G}-{(speed+speed_step/2)*2.24:.2G} mph @ |roll| < {max_abs_roll:.2G}")
    else
      title!(p[si,plot_col_num], f"{(speed-speed_step/2)*2.24:.2G}-{(speed+speed_step/2)*2.24:.2G} mph w/ |roll| < {max_abs_roll:.2G}")
    end
    xlabel!(p[si,plot_col_num], "lateral acceleration (m/s²; [+] = right turn)")
    ylabel!(p[si,plot_col_num], "steer command\n([+] = pushing right)")
  end

  # Iterate over the speed range and create a plot for each speed

  plot_col_num += 1

  # now w.r.t. lateral gravitational acceleration

  for (si, speed) in enumerate(speed_range)

    # Plot the training data
    X_train_filtered, y_train_filtered = filter_data_by_speed(X_train_rescaled, y_train, speed, speed_step/2, no_jerk=true)
    plot_scatter_step = round(Int, max(1, size(X_train_filtered, 1) / scatter_points_desired / 2))
    scatter!(p[si,plot_col_num], X_train_filtered[1:plot_scatter_step:end, 2], y_train_filtered[1:plot_scatter_step:end], label="Training Data", markersize=2, markercolor=train_color, markeralpha=marker_alpha, xlims=(-3.5, 3.5), ylims=(-1.4,1.4), markerstrokewidths=0)

    # Plot the test data
    X_test_filtered, y_test_filtered = filter_data_by_speed(X_test_rescaled, y_test, speed, speed_step/2, no_jerk=true)
    plot_scatter_step = round(Int, max(1, size(X_test_filtered, 1) / scatter_points_desired))
    scatter!(p[si,plot_col_num], X_test_filtered[1:plot_scatter_step:end, 2], y_test_filtered[1:plot_scatter_step:end], label="Test Data", markersize=2, markercolor=test_color, markeralpha=test_alpha, xlims=(-3.5, 3.5), ylims=(-1.4,1.4), markerstrokewidths=0)

    vline!(p[si,plot_col_num], [0.0], color=:black, linewidth=1, label="")
    hline!(p[si,plot_col_num], [0.0], color=:black, linewidth=1, label="")
    hline!(p[si,plot_col_num], [-1, 1], color=:red, linewidth=1, label="")

    # Plot the model output
    ci = 1
    for gla in -0.1:0.05:0.1
      x_model = Vector{Float32}(undef, length(lateral_acceleration_range))
      y_model = Vector{Float32}(undef, length(lateral_acceleration_range))
      
      for (i, la) in enumerate(lateral_acceleration_range)
          fill!(lat_accels, la)
          fill!(rolls, gla)
          
          input_data = [speed la 0.0 gla]
          input_data = hcat(input_data, lat_accels', rolls')
          steer_command = feedforward_function(input_data)
          
          x_model[i] = la
          y_model[i] = steer_command
      end
      
      plot!(p[si,plot_col_num], x_model, y_model, label="roll = $gla", linewidth=line_width, xlims=(-3.5, 3.5), ylims=(-1.4,1.4), color=palette(cpalette,5)[ci])
      ci += 1
    end

    # Configure the plot's appearance
    if si == 1
      title!(p[1,plot_col_num], f"Model test loss: {test_loss:.2G}\nRoll compensation [+] = leaning to the right\n{(speed-speed_step/2)*2.24:.2G}-{(speed+speed_step/2)*2.24:.2G} mph w/ |lat jerk| < {max_abs_lat_jerk:.2G}")
    else
      title!(p[si,plot_col_num], f"{(speed-speed_step/2)*2.24:.2G}-{(speed+speed_step/2)*2.24:.2G} mph w/ |lat jerk| < {max_abs_lat_jerk:.2G}")
    end
    xlabel!(p[si,plot_col_num], "lateral acceleration (m/s²; [+] = right turn)")
    ylabel!(p[si,plot_col_num], "steer command\n([+] = pushing right)")
  end


  # Display the plot
  savefig(p, "$plot_path/$car_name-a.png")
  savefig(p, "$plot_path/$car_name-a.pdf")
  # display(p)

  # Now plot model response to error, lateral jerk, and roll.
  # Each row will be different speeds as above. The left column will show the model output
  # as a function of error (instantaneous lateral jerk) at different lateral accels (as different lines). 
  # The second column will show model output as a function of lateral jerk (i.e. past/future lateral accels).
  # The third column will be as a function of *constant* roll (all past/future the same), and the 
  # fourth column will be as a function of dynamic roll (past/future changing linearly).

  plot_col_num = 1
  lateral_accel_range = -2.0:1.0:2.0
  lateral_jerk_range = -3.0:0.1:3.0
  roll_range = -0.2:0.01:0.2

  p = plot(layout = (size(collect(speed_range), 1), 4), legend=:bottomright, size=(2500, 2300), margin=8mm)

  # first w.r.t. error
  for (plot_row_num, speed) in enumerate(speed_range)
    vline!(p[plot_row_num,plot_col_num], [0.0], color=:black, linewidth=1, label="")
    hline!(p[plot_row_num,plot_col_num], [0.0], color=:black, linewidth=1, label="")
    hline!(p[plot_row_num,plot_col_num], [-1, 1], color=:red, linewidth=1, label="")

    # Plot the model output
    ci = 1
    for la in lateral_accel_range
      x_model = Vector{Float32}(undef, length(lateral_jerk_range))
      y_model = Vector{Float32}(undef, length(lateral_jerk_range))
      
      # Pre-fill lat_accels with constant values
      fill!(lat_accels, la)
      fill!(rolls, 0f0)
      
      for (i, lj) in enumerate(lateral_jerk_range)
          input_data = [speed la lj 0.0]
          input_data = hcat(input_data, lat_accels', rolls')
          steer_command = feedforward_function(input_data)
          x_model[i] = lj
          y_model[i] = steer_command
      end
      
      plot!(p[plot_row_num,plot_col_num], x_model, y_model, label="lat accel = $la", linewidth=line_width, xlims=(-3.0, 3.0), ylims=(-1.5,1.5), color=palette(cpalette,5)[ci])
      ci += 1
    end

    # Configure the plot's appearance
    if plot_row_num == 1
      title!(p[1,plot_col_num], f"Error response\n{(speed-speed_step/2)*2.24:.2G}-{(speed+speed_step/2)*2.24:.2G} mph")
    else
      title!(p[plot_row_num,plot_col_num], f"{(speed-speed_step/2)*2.24:.2G}-{(speed+speed_step/2)*2.24:.2G} mph")
    end
    xlabel!(p[plot_row_num,plot_col_num], "lateral acceleration/jerk error (m/s²; [+] = correcting to the right)")
    ylabel!(p[plot_row_num,plot_col_num], "steer command\n([+] = pushing right)")
  end

  # now plot model response to lateral jerk
  plot_col_num += 1
  for (plot_row_num, speed) in enumerate(speed_range)
    vline!(p[plot_row_num,plot_col_num], [0.0], color=:black, linewidth=1, label="")
    hline!(p[plot_row_num,plot_col_num], [0.0], color=:black, linewidth=1, label="")
    hline!(p[plot_row_num,plot_col_num], [-1, 1], color=:red, linewidth=1, label="")

    # Plot the model output
    ci = 1
    for la in lateral_accel_range
      x_model = Vector{Float32}(undef, length(lateral_jerk_range))
      y_model = Vector{Float32}(undef, length(lateral_jerk_range))
      
      for (i, lj) in enumerate(lateral_jerk_range)
          # Calculate lat_accels with lateral jerk
          @inbounds for (idx, t) in enumerate(t_list)
              lat_accels[idx] = la + lj * t
              rolls[idx] = 0f0
          end
          
          input_data = [speed la 0.0 0.0]
          input_data = hcat(input_data, lat_accels', rolls')
          steer_command = feedforward_function(input_data)
          
          x_model[i] = lj
          y_model[i] = steer_command
      end
      
      plot!(p[plot_row_num,plot_col_num], x_model, y_model, label="lat accel = $la", linewidth=line_width, xlims=(-3.0, 3.0), ylims=(-1.5,1.5), color=palette(cpalette,5)[ci])
      ci += 1
    end

    # Configure the plot's appearance
    if plot_row_num == 1
      title!(p[1,plot_col_num], f"{car_name}\nLateral jerk response\n{(speed-speed_step/2)*2.24:.2G}-{(speed+speed_step/2)*2.24:.2G} mph")
    else
      title!(p[plot_row_num,plot_col_num], f"{(speed-speed_step/2)*2.24:.2G}-{(speed+speed_step/2)*2.24:.2G} mph")
    end
    xlabel!(p[plot_row_num,plot_col_num], "lateral jerk (m/s²; [+] = wheel moving to the right)")
    ylabel!(p[plot_row_num,plot_col_num], "steer command\n([+] = pushing right)")
  end

  # now plot model response to constant roll
  plot_col_num += 1
  for (plot_row_num, speed) in enumerate(speed_range)
    vline!(p[plot_row_num,plot_col_num], [0.0], color=:black, linewidth=1, label="")
    hline!(p[plot_row_num,plot_col_num], [0.0], color=:black, linewidth=1, label="")
    hline!(p[plot_row_num,plot_col_num], [-1, 1], color=:red, linewidth=1, label="")

    # Plot the model output
    ci = 1
    for la in lateral_accel_range
      x_model = Vector{Float32}(undef, length(roll_range))
      y_model = Vector{Float32}(undef, length(roll_range))
      
      # Pre-fill lat_accels with constant values
      fill!(lat_accels, la)
      
      for (i, ro) in enumerate(roll_range)
          # Fill rolls with constant value
          fill!(rolls, ro)
          
          input_data = [speed la 0.0 ro]
          input_data = hcat(input_data, lat_accels', rolls')
          steer_command = feedforward_function(input_data)
          
          x_model[i] = ro
          y_model[i] = steer_command
      end
      
      plot!(p[plot_row_num,plot_col_num], x_model, y_model, label="lat accel = $la", linewidth=line_width, xlims=(-0.2, 0.2), ylims=(-1.5,1.5), color=palette(cpalette,5)[ci])
      ci += 1
    end

    # Configure the plot's appearance
    if plot_row_num == 1
      title!(p[1,plot_col_num], f"Roll response\n{(speed-speed_step/2)*2.24:.2G}-{(speed+speed_step/2)*2.24:.2G} mph")
    else
      title!(p[plot_row_num,plot_col_num], f"{(speed-speed_step/2)*2.24:.2G}-{(speed+speed_step/2)*2.24:.2G} mph")
    end
    xlabel!(p[plot_row_num,plot_col_num], "Road roll (radians; [+] = leaning right)")
    ylabel!(p[plot_row_num,plot_col_num], "steer command\n([+] = pushing right)")
  end

  # now plot model response to roll rate (same as previous, but
  # lateral acceleration lines are replaced with lines with different roll rates)
  roll_rate_range = -0.4:0.2:0.4
  plot_col_num += 1
  for (plot_row_num, speed) in enumerate(speed_range)
    vline!(p[plot_row_num,plot_col_num], [0.0], color=:black, linewidth=1, label="")
    hline!(p[plot_row_num,plot_col_num], [0.0], color=:black, linewidth=1, label="")
    hline!(p[plot_row_num,plot_col_num], [-1, 1], color=:red, linewidth=1, label="")

    # Plot the model output
    ci = 1
    for rr in roll_rate_range
      x_model = Vector{Float32}(undef, length(roll_range))
      y_model = Vector{Float32}(undef, length(roll_range))
      
      # Pre-fill lat_accels with zeros
      fill!(lat_accels, 0f0)
      
      for (i, ro) in enumerate(roll_range)
          # Calculate rolls with roll rate
          @inbounds for (idx, t) in enumerate(t_list)
              rolls[idx] = ro + t * rr
          end
          
          input_data = [speed 0.0 0.0 ro]
          input_data = hcat(input_data, lat_accels', rolls')
          steer_command = feedforward_function(input_data)
          
          x_model[i] = ro
          y_model[i] = steer_command
      end
      
      plot!(p[plot_row_num,plot_col_num], x_model, y_model, label="roll rate = $rr", linewidth=line_width, xlims=(-0.2, 0.2), ylims=(-1.5,1.5), color=palette(cpalette,5)[ci])
      ci += 1
    end

    # Configure the plot's appearance
    if plot_row_num == 1
      title!(p[1,plot_col_num], f"Roll rate [rad/s] response\n{(speed-speed_step/2)*2.24:.2G}-{(speed+speed_step/2)*2.24:.2G} mph")
    else
      title!(p[plot_row_num,plot_col_num], f"{(speed-speed_step/2)*2.24:.2G}-{(speed+speed_step/2)*2.24:.2G} mph")
    end
    xlabel!(p[plot_row_num,plot_col_num], "Road roll (radians; [+] = leaning right)")
    ylabel!(p[plot_row_num,plot_col_num], "steer command\n([+] = pushing right)")
  end

  # Display the plot
  savefig(p, "$plot_path/$car_name-b.png")
  savefig(p, "$plot_path/$car_name-b.pdf")
end

function multiline_string(strings::Vector{String}, n::Int; prefix="")::String
  # Pre-allocate for better performance
  lines = Vector{String}()
  sizehint!(lines, div(length(strings), 5) + 1)  # Estimate number of lines needed
  
  current_line = prefix
  
  for str in strings
      # If the current string fits on the current line, add it to the line
      if length(current_line) + length(str) + 2 <= n
          if isempty(current_line)
              current_line = str
          else
              current_line *= ", $str"
          end
      # Otherwise, start a new line with the current string
      else
          push!(lines, current_line)
          current_line = str
      end
  end
  
  # Add the last line to the list
  if !isempty(current_line)
      push!(lines, current_line)
  end
  
  # Join the lines with newline characters
  return join(lines, ",\n")
end

function create_model(in_file, out_dir_base)
  carname = replace(Base.basename(in_file), ".feather" => "")
  outdir = create_folder_with_iterator(out_dir_base, carname, make_new=false)
  logfile = open(outdir * "/$(carname)_log.txt", "a")  # Open log file in append mode
  out_streams = TeeStream(stdout, logfile)  # Create a Tee output stream
  preprocess_infile = replace(in_file, ".feather" => "_balanced.feather")
  use_existing_input = false
  if isfile(preprocess_infile) # && stat(in_file).mtime < stat(preprocess_infile).mtime
      #use_existing_input = true
      # return
  end
  
  data = load_data(in_file, use_existing_input, outdir, out_streams)

  model_file = "$outdir/$carname.bson"
  use_existing_input = false
  println(out_streams, "Model file: $model_file")
  if isfile(model_file) && stat(in_file).mtime < stat(model_file).mtime
      use_existing_input = true
      println(out_streams, "Using existing model file: $model_file")
      # return
  end

  model, input_mean, input_std, X_train, y_train, X_test, y_test, test_loss = train_model(outdir, use_existing_input, data, out_streams)
  
  test_plot_model(model, outdir, X_train, y_train, X_test, y_test, input_mean, input_std, multiline_string(names(select(data, Not([:steer_cmd]))), 60, prefix="Model input: "), out_streams, test_loss)
  close(logfile)
end

function main(in_dir)
  # Get all feather files that aren't balanced files
  feather_files = filter(file -> occursin(".feather", file) && !occursin("_balanced.feather", file), readdir(in_dir))
  
  # Process each file
  for in_file in feather_files
      println("Processing $in_file")
      create_model(joinpath(in_dir, in_file), in_dir)
  end
end

# Use the current user's home directory instead of hardcoding
home_dir = ENV["HOME"]
main("$home_dir/Downloads/rlogs/output/GENESIS")
