import Pkg
# Pkg.add("CSV")
# Pkg.add("DataFrames")
# Pkg.add("StatsBase")
# Pkg.add("MultivariateStats")
# Pkg.add("Flux")
# Pkg.add("MLDataUtils")
# Pkg.add("PyFormattedStrings")
# Pkg.add("ProgressMeter")
# Pkg.add("Optim")
# Pkg.add("FluxOptTools")
# Pkg.add("BSON")
# Pkg.add("CategoricalArrays")
# Pkg.add("InvertedIndices")

using CSV
using DataFrames
using StatsBase
using MultivariateStats
using Flux
using Flux: params, train!, mse
using Flux.Data: DataLoader
using DataFrames
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
using CUDA
using ArgParse


function create_folder_with_iterator(path::AbstractString, folder_name::AbstractString)
  full_path = joinpath(path, folder_name)
  i = 1
  while isdir(full_path)
      folder_name_with_iterator = string(folder_name, "_", i)
      full_path = joinpath(path, folder_name_with_iterator)
      i += 1
  end
  mkdir(full_path)
  return full_path
end

function describe(arr)
  n = length(arr)
  μ = mean(arr)
  σ = std(arr)
  minimum_value = minimum(arr)
  maximum_value = maximum(arr)
  quartiles = quantile(arr, [0.25, 0.5, 0.75])
  
  return f"n: {n}, mean: {μ:0.3f}, std: {σ:0.3f}, min: {minimum_value:0.3f}, max: {maximum_value:0.3f}, 25%: {quartiles[1]:0.3f}, 50%: {quartiles[2]:0.3f}, 75%: {quartiles[3]:0.3f}"

end

function load_data(infile::String, use_existing_data::Bool, outdir::String)::DataFrame
  # Load the data into a DataFrame
  if use_existing_data
    # infile = replace(infile, ".csv" => "_balanced.csv")
    infile = replace(infile, ".feather" => "_balanced.feather")

    println("Using existing data")
  end
  # data = CSV.read(infile, DataFrame)
  data = Feather.read(infile)
  if !use_existing_data
    println("Loading data...")
    # Remove rows with missing data
    data = data[completecases(data), :]

    println(f"Loaded {nrow(data)} rows")

    # println(f"Loaded data: {names(data)}")
    println(f"Data {data[1:5, :]}")
    for col in names(data)
      if typeof(data[1, col]) == Float64
        println("$col: $(describe(collect(data[:,col])))")
      end
    end

    # Save the data to a feather file
    # println("Saving data to feather file")
    # Feather.write(joinpath(dirname(infile), replace(infile, ".csv" => ".feather")), data)

    println("Filtering out extreme values")

    min_vego = minimum(data[!, :v_ego])

    # setup bin ranges
    mm_v_ego = [0.1, 45.0]
    mm_steer_cmd = [-2.0, 2.0]
    mm_lateral_accel = [-4.1, 4.1]
    mm_lateral_jerk = [-5.0, 5.0]
    mm_roll = [-0.20, 0.20]

    # setup bins
    nbins = 41
    # this is the max number of points that will go in each bin. it's low because lowspeed data is scarse.
    sample_size = 200
    step_v_ego = (mm_v_ego[2] - mm_v_ego[1]) / nbins
    step_steer_cmd = (mm_steer_cmd[2] - mm_steer_cmd[1]) / nbins
    step_lateral_accel = (mm_lateral_accel[2] - mm_lateral_accel[1]) / nbins
    step_lateral_jerk = (mm_lateral_jerk[2] - mm_lateral_jerk[1]) / nbins
    step_roll = (mm_roll[2] - mm_roll[1]) / nbins


    # filter data
    old_nrows = nrow(data)
    println(f"{old_nrows} rows before filtering")
    data = filter(row -> mm_v_ego[1] < row.v_ego < mm_v_ego[2], data)
    println(f"Filtered out {old_nrows - nrow(data)} points with v_ego outside [{mm_v_ego[1]}, {mm_v_ego[2]}]")
    old_nrows = nrow(data)
    data = filter(row -> abs(row.steer_cmd) <= mm_steer_cmd[2], data)
    println(f"Filtered out {old_nrows - nrow(data)} points with steer_cmd outside [{-mm_steer_cmd[2]}, {mm_steer_cmd[2]}]")
    old_nrows = nrow(data)
    data = filter(row -> abs(row.lateral_accel) < mm_lateral_accel[2], data)
    println(f"Filtered out {old_nrows - nrow(data)} points with lateral_accel outside [{-mm_lateral_accel[2]}, {mm_lateral_accel[2]}]")
    old_nrows = nrow(data)
    data = filter(row -> abs(row.lateral_jerk) < mm_lateral_jerk[2], data)
    println(f"Filtered out {old_nrows - nrow(data)} points with lateral_jerk outside [{-mm_lateral_jerk[2]}, {mm_lateral_jerk[2]}]")
    old_nrows = nrow(data)
    data = filter(row -> abs(row.roll) < mm_roll[2], data)
    println(f"Filtered out {old_nrows - nrow(data)} points with roll outside [{-mm_roll[2]}, {mm_roll[2]}]")
    # data = filter(row -> abs(row.a_ego) < 2.95, data)
    # select!(data, Not([:a_ego])) # remove a_ego because it's not used in the model; no good correlations found
    println(f"{nrow(data)} rows after filtering")

    for col in names(data)
      if typeof(data[1, col]) == Float64
        println("$col: $(describe(collect(data[:,col])))")
      end
    end

    println(f"Calculating bins")
    data[!, :v_ego_bins] = cut(data[!, :v_ego], mm_v_ego[1]:step_v_ego:mm_v_ego[2])
    data[!, :lateral_accel_bins] = cut(data[!, :lateral_accel], mm_lateral_accel[1]:step_lateral_accel:mm_lateral_accel[2])
    data[!, :roll_bins] = cut(data[!, :roll], mm_roll[1]:step_roll:mm_roll[2])
    data[!, :lateral_jerk_bins] = cut(data[!, :lateral_jerk], mm_lateral_jerk[1]:step_lateral_jerk:mm_lateral_jerk[2])

    # create a combined column for balancing
    data[!,:combined_column] = string.(data[!,:v_ego_bins], "_", data[!,:lateral_accel_bins], "_", data[!,:roll_bins])
    # binning on all four variables is too sparse
    # data[!,:combined_column] = string.(data[!,:v_ego_bins], "_", data[!,:lateral_accel_bins], "_", data[!,:lateral_jerk_bins], "_", data[!,:roll_bins])
    
    # data = data[sample(1:nrow(data), 50000), :] # for testing

    # balance the data in parallel
    bin_dfs = Vector{Vector{DataFrame}}(undef, Threads.nthreads())
    for i in 1:Threads.nthreads()
        bin_dfs[i] = Vector{DataFrame}()
    end

    unique_bins = unique(data[!, :combined_column])
    prog = ProgressMeter.Progress(length(unique_bins), 1, "Balancing bins:")
    println(f"Balancing data into {length(unique_bins)} bins")
    for bin_label in unique_bins
        # println(f"Balancing bin {i} of {length(unique_bins)}")
        bin_data = data[data[!, :combined_column] .== bin_label, :]
        # m = Statistics.mean(bin_data[!, :steer_cmd])
        # s = Statistics.std(bin_data[!, :steer_cmd])
        # inliers = filter(row -> abs(row.steer_cmd - m) < 1.5*s, bin_data)
        # if nrow(inliers) > 0
        #   sampled_bin_data = inliers[sample(1:nrow(inliers), min(sample_size, nrow(inliers))), :]
        #   push!(bin_dfs[Threads.threadid()], sampled_bin_data)
        # else
        #   sampled_bin_data = bin_data[sample(1:nrow(bin_data), min(sample_size, nrow(bin_data))), :]
        #   push!(bin_dfs[Threads.threadid()], sampled_bin_data)
        # end


        sampled_bin_data = bin_data[sample(1:nrow(bin_data), min(sample_size, nrow(bin_data))), :]
        push!(bin_dfs[Threads.threadid()], sampled_bin_data)
        
        next!(prog)
    end
    flattened_sampled_bin_data = SplitApplyCombine.flatten(bin_dfs)
    data = vcat(flattened_sampled_bin_data...)
    # data = balanced_data

    bin_sizes = [size(i,1) for i in flattened_sampled_bin_data]
    println(f"Num points after initial balancing: {nrow(data)}")
    bin_min_size = minimum(bin_sizes)
    bin_max_size = maximum(bin_sizes)
    bin_mean_size = mean(bin_sizes)
    bin_std_size = std(bin_sizes)
    println(f"Bin sizes: min={bin_min_size}, max={bin_max_size}, mean={bin_mean_size}, std={bin_std_size}")


    new_bin_size = Int(round(max((bin_min_size + bin_mean_size)/2, mean(bin_sizes) - bin_std_size/2), digits=0))
    println(f"Shrinking bins to {new_bin_size} points")
    flattened_sampled_bin_data = [i[sample(1:size(i,1), new_bin_size), :] for i in flattened_sampled_bin_data]
    data = vcat(flattened_sampled_bin_data...)
    println(f"Num points after final balancing: {nrow(data)}")

    for col in names(data)
      if typeof(data[1, col]) == Float64
        println("$col: $(describe(collect(data[:,col])))")
      end
    end

    # CSV.write(joinpath(dirname(infile), replace(infile, ".csv" => "_balanced.csv")), data)
    Feather.write(joinpath(outdir, replace(infile, ".feather" => "_balanced.feather")), data)
  else
    println("Loading preprocessed data...")
  end

  return data
end

function train_model(working_dir::String, use_existing_model::Bool, data::DataFrame)::NamedTuple{(:model, :input_mean, :input_std, :X_train, :y_train, :X_test, :y_test), Tuple{Flux.Chain, Matrix{Float32}, Matrix{Float32}, Matrix{Float32}, Vector{Float32}, Matrix{Float32}, Vector{Float32}}}
  model_path = joinpath(working_dir, Base.basename(working_dir))

  # temp flip sign of roll
  data[!, :roll] = -data[!, :roll]

  # split into train and test sets
  train, test = stratifiedobs(row->row[:combined_column], data, p = 0.8)

  # remove columns used only for binning
  select!(data, Not([:combined_column]))
  select!(data, Not([:v_ego_bins]))
  select!(data, Not([:lateral_accel_bins]))
  select!(data, Not([:lateral_jerk_bins]))
  select!(data, Not([:roll_bins]))
  
  # split into independent an dependent variables
  X = Matrix(select(data, Not([:steer_cmd])))  # Assuming the last column is the dependent_var
  y = data[:, :steer_cmd];

  # Calculate the mean and standard deviation of the input features
  input_mean = mean(X, dims=1)
  input_std = std(X, dims=1)

  # Create a copy of the DataFrames with the signs of steer_cmd, lateral_accel, lateral_jerk, and roll reversed to make the data symmetric
  old_size = size(train, 1)
  println("Training data before copying symmmetric data: $old_size")
  data_sym = deepcopy(train)
  data_sym[!, :steer_cmd] = -1 .* data_sym[!, :steer_cmd]
  data_sym[!, :lateral_accel] = -1 .* data_sym[!, :lateral_accel]
  data_sym[!, :lateral_jerk] = -1 .* data_sym[!, :lateral_jerk]
  data_sym[!, :roll] = -1 .* data_sym[!, :roll]
  train = vcat(train, data_sym)
  println("Training data after copying symmmetric data: $(size(train,1))")

  old_size = size(test, 1)
  println("Test data before copying symmmetric data: $old_size")
  data_sym = deepcopy(test)
  data_sym[!, :steer_cmd] = -1 .* data_sym[!, :steer_cmd]
  data_sym[!, :lateral_accel] = -1 .* data_sym[!, :lateral_accel]
  data_sym[!, :lateral_jerk] = -1 .* data_sym[!, :lateral_jerk]
  data_sym[!, :roll] = -1 .* data_sym[!, :roll]
  test = vcat(test, data_sym)
  println("Test data after copying symmmetric data: $(size(test,1))")

  device = CUDA.functional() ? gpu : cpu
  CUDA.allowscalar(false)
  println("Using device: $(device)")
  # device = cpu

  # normalize the data
  X_train = Matrix(select(train, Not([:steer_cmd])))
  # println("$(train[1, :])")
  # println("as matrix: $(X_train[1, :])")
  X_train = (X_train .- input_mean) ./ input_std
  X_train = Array{Float32}(X_train) |> device
  # println("normalized $(X_train[1, :])")
  y_train = train[:, "steer_cmd"]
  y_train = Array{Float32}(y_train) |> device
  # println("steer cmd: $(y_train[1])")
  X_test = Matrix(select(test, Not([:steer_cmd])))
  X_test = (X_test .- input_mean) ./ input_std
  X_test = Array{Float32}(X_test) #|> device
  y_test = test[:, "steer_cmd"]
  y_test = Array{Float32}(y_test) #|> device

  input_dim = size(X_train, 2)

  # Define the model
  model = Chain(
      Dense(input_dim, 8, sigmoid),
      Dense(8, 16, sigmoid),
      Dense(16, 16),
      Dense(16, 1)
  ) |> device

  # Define the loss function, which includes penalties to enforce physically correct behavior

  # Define the range of values for each independent variable
  v_ego_range = range(1, stop=40, length=14)
  lateral_acceleration_range = range(-4, stop=4, length=11)
  lateral_acceleration_range_hi = range(-3.99, stop=4.01, length=11)
  lateral_jerk_range = range(-3, stop=3, length=15)
  lateral_jerk_range_hi = range(-2.99, stop=3.01, length=15)
  roll_range = range(-0.2, stop=0.2, length=21)
  roll_range_hi = range(-0.195, stop=0.205, length=21)

  # # Create a regular grid of points using Iterators.product
  grid = hcat([(collect(x) .- input_mean) ./ input_std for x in Iterators.product(v_ego_range, lateral_acceleration_range, lateral_jerk_range, roll_range)]...) |> device
  grid_da = hcat([(collect(x) .- input_mean) ./ input_std for x in Iterators.product(v_ego_range, lateral_acceleration_range_hi, lateral_jerk_range, roll_range)]...) |> device
  grid_dj = hcat([(collect(x) .- input_mean) ./ input_std for x in Iterators.product(v_ego_range, lateral_acceleration_range, lateral_jerk_range_hi, roll_range)]...) |> device
  grid_dg = hcat([(collect(x) .- input_mean) ./ input_std for x in Iterators.product(v_ego_range, lateral_acceleration_range, lateral_jerk_range, roll_range_hi)]...) |> device
  grid_odd_neg = hcat([(collect(x) .- input_mean) ./ input_std for x in Iterators.product(v_ego_range, -lateral_acceleration_range, -lateral_jerk_range, -roll_range)]...) |> device

  # println(typeof(grid))
  # println(typeof(X_test))

  d_odd_eye_cpu = Matrix{Float32}(I, 4, 4)
  for i in 2:4
    d_odd_eye_cpu[i,i] = -1.0
  end

  d_odd_eye = device(d_odd_eye_cpu)

  # function shift_column(matrix::Matrix, column_index::Int, shift::Float32)
  #     rows, cols = size(matrix)
  #     if cols != 4
  #         error("The input matrix should have 4 columns.")
  #     end

  #     if column_index < 1 || column_index > 4
  #         error("Invalid column index. It should be between 1 and 4.")
  #     end

  #     # Create a Nx4 matrix with the shift value at the desired position
  #     shift_matrix = zeros(Float32, rows, cols)
  #     shift_matrix[:, column_index] .= shift

  #     # Perform matrix addition to shift the specified column
  #     return matrix + shift_matrix
  # end

  function physical_constraint_losses(x, y_pred, λ1, λ2)
      model_grid = model(grid)
      model_da = model(grid_da)
      model_dj = model(grid_dj)
      model_dg = model(grid_dg)
      model_odd_neg = model(grid_odd_neg)

      monotonicity_loss = sum(max.(0, (model_da .- model_grid) .* -1)) +
                          sum(max.(0, (model_dj .- model_grid) .* -1)) +
                          sum(max.(0, (model_dg .- model_grid) .* -1))

      odd_loss = sum(abs.(model_grid .+ model_odd_neg))

      # Apply the d_odd_eye transformation to x
      # transformed_x = (x' * d_odd_eye)'

      # # Compute the model output for transformed_x
      # transformed_y_pred = model(transformed_x)

      # odd_loss += sum(abs.(y_pred .+ transformed_y_pred))

      # Total loss with penalty weights λ1 and λ2
      # λ1 = 0.0002
      # λ2 = 0.00002

      # return λ2 * odd_loss
      return λ1 * monotonicity_loss + λ2 * odd_loss
  end

  function loss1(x, y, λ1, λ2) 
      y_pred = model(x)
      standard_loss = Flux.mse(y_pred, y')

      # return standard_loss

      total_loss = standard_loss + physical_constraint_losses(x, y_pred, λ1, λ2)

      return total_loss
  end

  # function loss(x, y)
  #     y_pred = model(x)
  #     standard_loss = Flux.mse(y_pred, y')

  #     return standard_loss
  # end

  function combined_loss(x, y_true, y_pred, model, λ, λ1, λ2)
      mse = Flux.Losses.mse(y_true, y_pred)
      l2 = sum(p -> sum(abs2, p), params(model)) * λ
      physical_constraints = physical_constraint_losses(x, y_pred, λ1, λ2)
      return mse + l2 + physical_constraints
  end

  loss(x, y, λ=0.0, λ1=0.0, λ2=0.00002) = combined_loss(x, y', model(x), model, λ, λ1, λ2)

  # loss(x, y) = Flux.mse(model(x), y')

  # pick an optimizer
  # opt = Flux.ADAM(0.001)
  # opt = Flux.Nesterov()
  opt = Flux.AdaGrad()

  # train the model (with batches of shuffled data)
  tol = log10(size(X_train, 1)) > 7 ? 1e-4 : 2e-6
  Δtol = log10(size(X_train, 1)) > 7 ? 1e-4 : 5e-5
  logstep = device == gpu ? 50 : log10(size(X_train, 1)) > 7 ? 3 : 10
  logstepgrowth = 1
  logstepfloat = Float32(logstep)
  logstepbig = 10
  Δloss = Inf
  ΔΔloss = Inf
  Δloss_last = 0.0
  loss_last = Inf
  loss_cur = 0.0
  stall_check_count = device == gpu ? 50 : 15
  stall_count = 0
  epoch = 1
  epoch_max = device == gpu ? 30000 : log10(size(X_train, 1)) > 6 ? 150 : 1000
  epoch_min = 25
  batch_size = 1000
  train_data_loader = DataLoader((X_train', y_train), batchsize=batch_size, shuffle=true)


  println(size(X_train))
  println(size(y_train))

  # old_model = "$model_path.bson" #"/Users/haiiro/NoSync/voltlat.bson"
  # println("Loading old model, $old_model")
  # @load old_model model

  # model = device(model)

  if use_existing_model
      old_model = "$model_path.bson" #"/Users/haiiro/NoSync/voltlat.bson"
      println("Loading old model, $old_model")
      @load old_model model
  else
    while epoch < epoch_min || ((abs(Δloss) > tol || abs(ΔΔloss) > Δtol) && epoch < epoch_max)
        l = 0.0
        if device == cpu
          for (X_batch, y_batch) in train_data_loader
            gs = Flux.gradient(params(model)) do 
              l = loss(X_batch, y_batch)
            end
            Flux.Optimise.update!(opt, params(model), gs)
          end
        else
          gs = Flux.gradient(params(model)) do 
            l = loss(X_train', y_train)
          end
          Flux.Optimise.update!(opt, params(model), gs)
        end
        
        if (epoch % logstep == 0 || epoch == 1)
            loss_cur = l
            Δloss = loss_cur - loss_last
            if Δloss > 0.0
                stall_count += 1
            end
            if stall_count ≥ stall_check_count && Δloss < 0.0
                println("Stalled at epoch $epoch, loss $loss_cur")
                break
            end
            ΔΔloss = Δloss - Δloss_last
            loss_last = loss_cur
            Δloss_last = Δloss
            if abs(Δloss) > tol || abs(Δloss_last) > Δtol
                c1 = abs(Δloss) > tol ? ">" : "≤"
                c2 = abs(ΔΔloss) > Δtol ? ">" : "≤"
                cur_time = Dates.format(now(), "HH:MM:SS")
                println(f"round 1 {cur_time} Epoch: {epoch:3d} (of {epoch_max}; {stall_count} stalls of {stall_check_count}), Loss: {loss_cur:.6f}, ΔLoss: {Δloss:.7f} {c1} {tol:.6G}, ΔΔLoss: {ΔΔloss:.9f} {c2} {Δtol:.6G}")
            end
            logstepfloat *= logstepgrowth
            logstep = round(Int, logstepfloat)
        end
        epoch += 1
    end

    Δloss = Inf
    ΔΔloss = Inf
    Δloss_last = 0.0
    loss_last = Inf
    loss_cur = 0.0
    stall_count = 0
    epoch = 1
    epoch_max = device == gpu ? 5000 : log10(size(X_train, 1)) > 6 ? 150 : 1000

    λ1max = 0.0002
    λ2max = 0.00002

    # while epoch < epoch_min || ((abs(Δloss) > tol || abs(ΔΔloss) > Δtol) && epoch < epoch_max)
    #     l = 0.0
    #     λ1 = λ1max * (epoch / (epoch_max * 0.5)) |> device
    #     λ2 = λ2max * (epoch / (epoch_max * 0.5)) |> device
    #     if device == cpu
    #       for (X_batch, y_batch) in train_data_loader
    #         gs = Flux.gradient(params(model)) do
    #           l = loss1(X_batch, y_batch, λ1, λ2)
    #         end
    #         Flux.Optimise.update!(opt, params(model), gs)
    #       end
    #     else
    #       gs = Flux.gradient(params(model)) do 
    #         l = loss1(X_train', y_train, λ1, λ2)
    #       end
    #       Flux.Optimise.update!(opt, params(model), gs)
    #     end

    #     λ1 = λ1 |> cpu
    #     λ2 = λ2 |> cpu
        
    #     if (epoch % logstep == 0 || epoch == 1)
    #         loss_cur = l
    #         Δloss = loss_cur - loss_last
    #         if Δloss > 0.0
    #             stall_count += 1
    #         end
    #         if stall_count ≥ stall_check_count && Δloss < 0.0
    #             println("Stalled at epoch $epoch, loss $loss_cur")
    #             break
    #         end
    #         ΔΔloss = Δloss - Δloss_last
    #         loss_last = loss_cur
    #         Δloss_last = Δloss
    #         if abs(Δloss) > tol || abs(Δloss_last) > Δtol
    #             c1 = abs(Δloss) > tol ? ">" : "≤"
    #             c2 = abs(ΔΔloss) > Δtol ? ">" : "≤"
    #             cur_time = Dates.format(now(), "HH:MM:SS")
    #             println(f"round 2 {cur_time} Epoch: {epoch:3d} (of {epoch_max} with loss coeffs {λ1:e}, {λ2:e}; {stall_count} stalls of {stall_check_count}), Loss: {loss_cur:.6f}, ΔLoss: {Δloss:.6f} {c1} {tol:.6G}, ΔΔLoss: {ΔΔloss:.6f} {c2} {Δtol:.6G}")
    #         end
    #         logstepfloat *= logstepgrowth
    #         logstep = round(Int, logstepfloat)
    #     end
    #     epoch += 1
    # end
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
      grid_dg = cpu(grid_dg)
      grid_odd_neg = cpu(grid_odd_neg)
      d_odd_eye = cpu(d_odd_eye)
    end

    @save "$model_path.bson" model # "/Users/haiiro/NoSync/voltlat.bson" model
    println("Finished after $epoch epochs, Loss: $loss_cur, ΔLoss: $Δloss, Test loss: $(loss(X_test', y_test))")

  end

  function feedforward_function(input_data; zero_bias=false)
    # Scale the input data using the stored mean and standard deviation values
    input_data_scaled = (input_data .- input_mean) ./ input_std
    if zero_bias
      eval_model = deepcopy(model)
      for layer in eval_model.layers
        if layer isa Dense
          Flux.params(layer)[2] .= zeros(size(Flux.params(layer)[2]))
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
        b = zeros(size(b))
      end
      # println("$(size(W)), $(size(b)), $(layer.σ), $(size(x))")
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
          println("Unsupported activation function: $(layer.σ)")
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
    vego_range = 0:20:40
    lataccel_range = -4:4:4
    latjerk_range = -3:3:3
    roll_range = -0.2:0.2:0.2
    println("Testing manual model evaluation (as performed in OpenPilot)...")
    println("Testing with zero bias: $zero_bias")
    test_dict = Dict()
    for vego in vego_range
      for lataccel in lataccel_range
        for latjerk in latjerk_range
          for roll in roll_range
            x = Float32[vego lataccel latjerk roll]
            result_model = feedforward_function(x, zero_bias=zero_bias)  # Model evaluation
            result_manual = feedforward_function_manual(x, zero_bias=zero_bias)  # Manual evaluation
            test_dict[f"[{x[1]}, {x[2]}, {x[3]}, {x[4]}]"] = result_model
            if !isapprox(result_model, result_manual, atol=1e-6)
              println("Mismatch at input: $x")
              println("Model: $result_model, Manual: $result_manual")
              return false
            end
          end
        end
      end
    end

    println("Test passed: All outputs match!")
    return test_dict
  end


  # Example input (v_ego	lateral_accel	lateral_jerk g_lat_accel)
  example_input = [25 0.5 0.1 0.05]
  steer_command = feedforward_function(example_input)
  println("Steer command @ $example_input: ", steer_command)
  steer_command = feedforward_function_manual(example_input)
  println("Steer manual command @ $example_input: ", steer_command)

  test_dict_zero_bias = test_evaluate_manually(model, zero_bias=true)
  test_dict = test_evaluate_manually(model)

  current_date_and_time = Dates.format(now(), "yyyy-mm-dd_HH-MM-SS")

  model_test_loss = loss(X_test', y_test)

  # save model to json for Python import
  function export_model_params_to_json(model::Chain, input_mean::Matrix{Float32}, input_std::Matrix{Float32}, filename::String, test_dict, test_dict_zero_bias, current_date_and_time, model_test_loss)
      W, b = params(model.layers[1])
      input_size = size(W, 2)
      output_size = size(params(model.layers[end])[1], 1)
      params_dict = Dict{String, Any}("input_size" => input_size, "output_size" => output_size, "layers" => [], "input_mean" => input_mean, "input_std" => input_std, "test_dict" => test_dict, "test_dict_zero_bias" => test_dict_zero_bias, "current_date_and_time" => current_date_and_time, "model_test_loss" => model_test_loss)

      for (idx, layer) in enumerate(model.layers)
          if isa(layer, Dense)
              W, b = params(layer)
              params_dict["layers"] = push!(params_dict["layers"], Dict(
                  "dense_$(idx)_W" => Array(W'),
                  "dense_$(idx)_b" => Array(b'),
                  "activation" => string(layer.σ)
              ))
          end
      end

      open(filename, "w") do f
          write(f, JSON.json(params_dict))
      end
  end

  export_model_params_to_json(model, Matrix{Float32}(input_mean), Matrix{Float32}(input_std), "$model_path.json", test_dict, test_dict_zero_bias, current_date_and_time, model_test_loss)


  # Evaluate the model on the test set 
  test_loss = loss(X_test', y_test)
  println("Test loss (MSE): ", test_loss)

  return (model=model, input_mean=input_mean, input_std=input_std, X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test)

end

function test_plot_model(model::Flux.Chain, plot_path::String, X_train::Matrix{Float32}, y_train::Vector{Float32}, X_test::Matrix{Float32}, y_test::Vector{Float32}, input_mean::Matrix{Float32}, input_std::Matrix{Float32})

  car_name = Base.basename(plot_path)

  function feedforward_function(input_data)
    # Scale the input data using the stored mean and standard deviation values
    input_data_scaled = (input_data .- input_mean) ./ input_std
    steer_command = model(input_data_scaled')
    return steer_command[1]
  end

  max_abs_lat_jerk = 0.2
  max_abs_roll = 0.05

  # Create a function to filter the dataset based on speed
  function filter_data_by_speed(Xi, yi, speed, tolerance; no_jerk=false, no_roll=false)
    indices = findall(abs.(Xi[:, 1] .- speed) .< tolerance)
    X, y = Xi[indices, :], yi[indices]
    if no_jerk
      indices = findall(abs.(X[:, 3]) .< max_abs_lat_jerk)
      X, y = X[indices, :], y[indices]
    end
    if no_roll
      indices = findall(abs.(X[:, 4]) .< max_abs_roll)
      X, y = X[indices, :], y[indices]
    end
    return X, y
  end

  X_train_rescaled = X_train .* input_std .+ input_mean
  X_test_rescaled = X_test .* input_std .+ input_mean

  plot_scatter_step = round(Int, max(1, size(X_train, 1) / 10000))

  # Iterate over the speed range and create a plot for each speed
  # first w.r.t. lateral jerk
  speed_step = 6
  speed_range = 0:speed_step:30
  lateral_acceleration_range = range(-4.0, 4.0, length=100)

  plot_col_num = 1
  p = plot(layout = (size(collect(speed_range), 1), 2), legend=:bottomright, size=(1300, 2300), margin=10mm)

  for (si, speed) in enumerate(speed_range)
    # Plot the training data
    X_train_filtered, y_train_filtered = filter_data_by_speed(X_train_rescaled, y_train, speed, speed_step/2, no_roll=true)
    scatter!(p[si,plot_col_num], X_train_filtered[1:plot_scatter_step:end, 2], y_train_filtered[1:plot_scatter_step:end], label="Training Data", markersize=2, markercolor=:grey, markeralpha=0.1, xlims=(-3.5, 3.5), ylims=(-1.4,1.4))

    # Plot the test data
    X_test_filtered, y_test_filtered = filter_data_by_speed(X_test_rescaled, y_test, speed, speed_step/2, no_roll=true)
    scatter!(p[si,plot_col_num], X_test_filtered[1:plot_scatter_step:end, 2], y_test_filtered[1:plot_scatter_step:end], label="Test Data", markersize=2, markercolor=:green, markeralpha=0.2, xlims=(-3.5, 3.5), ylims=(-1.4,1.4))

    # Plot the model output
    for lj in -1:0.5:1
      x_model = []
      y_model = []
      for la in lateral_acceleration_range
          # (v_ego	lateral_accel	lateral_jerk g_lat_accel)
          input_data = [speed la lj 0.0]
          steer_command = feedforward_function(input_data)
          push!(x_model, la)
          push!(y_model, steer_command)
      end
      plot!(p, xticks=true, yticks=true)
      plot!(p[si,plot_col_num], x_model, y_model, label="lat_jerk = $lj", linewidth=4, xlims=(-3.5, 3.5), ylims=(-1.4,1.4))
    end


    # Configure the plot's appearance
    if si == 1
      title!(p[1,plot_col_num], f"{car_name}\nSteer vs. a_lat at {speed*2.24:.2G}-{(speed+speed_step)*2.24:.2G} @ |roll| < {max_abs_roll:.2G}")
    else
      title!(p[si,plot_col_num], f"{speed*2.24:.2G}-{(speed+speed_step)*2.24:.2G} mph")
    end
    ylabel!(p[si,plot_col_num], "Steer Command")
  end
  xlabel!(p[size(collect(speed_range),1),plot_col_num], "a_lat (m/s²)")

  # now w.r.t. lateral gravitational acceleration

  # Iterate over the speed range and create a plot for each speed

  plot_col_num += 1

  for (si, speed) in enumerate(speed_range)

    # Plot the training data
    X_train_filtered, y_train_filtered = filter_data_by_speed(X_train_rescaled, y_train, speed, speed_step/2, no_jerk=true)
    scatter!(p[si,plot_col_num], X_train_filtered[1:plot_scatter_step:end, 2], y_train_filtered[1:plot_scatter_step:end], label="Training Data", markersize=2, markercolor=:grey, markeralpha=0.1, xlims=(-3.5, 3.5), ylims=(-1.4,1.4))

    # Plot the test data
    X_test_filtered, y_test_filtered = filter_data_by_speed(X_test_rescaled, y_test, speed, speed_step/2, no_jerk=true)
    scatter!(p[si,plot_col_num], X_test_filtered[1:plot_scatter_step:end, 2], y_test_filtered[1:plot_scatter_step:end], label="Test Data", markersize=2, markercolor=:green, markeralpha=0.2, xlims=(-3.5, 3.5), ylims=(-1.4,1.4))

    # Plot the model output
    for gla in -0.1:0.05:0.1
      x_model = []
      y_model = []
      for la in lateral_acceleration_range
          # (v_ego	lateral_accel	lateral_jerk g_lat_accel)
          input_data = [speed la 0.0 gla]
          steer_command = feedforward_function(input_data)
          push!(x_model, la)
          push!(y_model, steer_command)
      end
      plot!(p[si,plot_col_num], x_model, y_model, label="roll = $gla", linewidth=4, xlims=(-3.5, 3.5), ylims=(-1.4,1.4))
    end

    # Configure the plot's appearance
    if si == 1
      title!(p[1,plot_col_num], f"Steer vs. a_lat at {speed*2.24:.2G}-{(speed+speed_step)*2.24:.2G} @ |lat jerk| < {max_abs_lat_jerk:.2G}")
    else
      title!(p[si,plot_col_num], f"{speed*2.24:.2G}-{(speed+speed_step)*2.24:.2G} mph")
    end
  end
  xlabel!(p[size(collect(speed_range),1),plot_col_num], "a_lat (m/s²)")
  # Display the plot
  savefig(p, "$plot_path/$car_name.png")
  display(p)
end

function create_model(in_file, out_dir_base)
  carname = replace(Base.basename(in_file), ".feather" => "")
  outdir = create_folder_with_iterator(out_dir_base, carname)
  preprocess_infile = replace(in_file, ".feather" => "_balanced.feather")
  use_existing_input = false
  if isfile(preprocess_infile) && stat(in_file).mtime < stat(preprocess_infile).mtime
      use_existing_input = true
      # return
  end
  
  data = load_data(in_file, use_existing_input, outdir)

  model, input_mean, input_std, X_train, y_train, X_test, y_test = train_model(outdir, false, data)
  
  test_plot_model(model, outdir, X_train, y_train, X_test, y_test, input_mean, input_std)
end

function main(in_dir)
  for in_file in readdir(in_dir)
    if occursin(".feather", in_file) && !occursin("_balanced.feather", in_file)
      println("Processing $in_file")
      create_model(joinpath(in_dir, in_file), in_dir)
      return
    end
  end
end

# parser = ArgParse.ArgParser("Train a model from a feather file of [vego, lat_accel, lat_jerk, roll, steer_torque] that will predict steer_torque from [vego, lat_accel, lat_jerk, roll]") 

# add_argument(parser, "--input-dir", help="Input directory full of feather files", required=true)

# args = parse_args(parser)

# main(args["input-dir"])

main("/mnt/video/scratch-video/latmodels")


