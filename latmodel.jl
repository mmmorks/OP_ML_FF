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
using CSV
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

use_existing_data = true
use_existing_model = true

function load_data(infile::String, use_existing_data::Bool)::DataFrame
  # Load the data into a DataFrame
  if use_existing_data
    infile = replace(infile, ".csv" => "_balanced.csv")
    println("Using existing data")
  end
  data = CSV.read(infile, DataFrame)
  if !use_existing_data
    # Remove rows with missing data
    data = data[completecases(data), :]

    println(f"Loaded {nrow(data)} rows")

    # setup bin ranges
    mm_v_ego = [2, 40]
    mm_steer_cmd = [-2.0, 2.0]
    mm_lateral_accel = [-4.45, 4.45]
    mm_lateral_accel_rate_no_roll = [-2.95, 2.95]
    mm_roll = [-2.95, 2.95]

    # setup bins
    nbins = 80
    # this is the max number of points that will go in each bin. it's low because lowspeed data is scarse.
    sample_size = 3 
    step_v_ego = (mm_v_ego[2] - mm_v_ego[1]) / nbins
    step_steer_cmd = (mm_steer_cmd[2] - mm_steer_cmd[1]) / nbins
    step_lateral_accel = (mm_lateral_accel[2] - mm_lateral_accel[1]) / nbins
    step_lateral_accel_rate_no_roll = (mm_lateral_accel_rate_no_roll[2] - mm_lateral_accel_rate_no_roll[1]) / nbins
    step_roll = (mm_roll[2] - mm_roll[1]) / nbins

    # filter data
    data = filter(row -> mm_v_ego[1] < row.v_ego < mm_v_ego[2], data)
    data = filter(row -> abs(row.steer_cmd) <= mm_steer_cmd[2], data)
    data = filter(row -> abs(row.lateral_accel) < mm_lateral_accel[2], data)
    data = filter(row -> abs(row.lateral_accel_rate_no_roll) < mm_lateral_accel_rate_no_roll[2], data)
    data = filter(row -> abs(row.roll) < mm_roll[2], data)
    # data = filter(row -> abs(row.a_ego) < 2.95, data)
    select!(data, Not([:a_ego])) # remove a_ego because it's not used in the model; no good correlations found
    println(f"Filtered to {nrow(data)} rows")


    println(f"Calculating bins")
    data[!, :v_ego_bins] = cut(data[!, :v_ego], mm_v_ego[1]:step_v_ego:mm_v_ego[2])
    data[!, :lateral_accel_bins] = cut(data[!, :lateral_accel], mm_lateral_accel[1]:step_lateral_accel:mm_lateral_accel[2])
    data[!, :roll_bins] = cut(data[!, :roll], mm_roll[1]:step_roll:mm_roll[2])
    data[!, :lateral_accel_rate_no_roll_bins] = cut(data[!, :lateral_accel_rate_no_roll], mm_lateral_accel_rate_no_roll[1]:step_lateral_accel_rate_no_roll:mm_lateral_accel_rate_no_roll[2])

    # create a combined column for balancing
    data[!,:combined_column] = string.(data[!,:v_ego_bins], "_", data[!,:lateral_accel_bins])
    # binning on all four variables is too sparse
    # data[!,:combined_column] = string.(data[!,:v_ego_bins], "_", data[!,:lateral_accel_bins], "_", data[!,:lateral_accel_rate_no_roll_bins], "_", data[!,:roll_bins])


    # balance the data in parallel
    bin_dfs = Vector{Vector{DataFrame}}(undef, Threads.nthreads())
    for i in 1:Threads.nthreads()
        bin_dfs[i] = Vector{DataFrame}()
    end

    unique_bins = unique(data[!, :combined_column])
    prog = ProgressMeter.Progress(length(unique_bins), 1, "Balancing bins:")
    println(f"Balancing data into {length(unique_bins)} bins")
    @threads for bin_label in unique_bins
        # println(f"Balancing bin {i} of {length(unique_bins)}")
        bin_data = data[data[!, :combined_column] .== bin_label, :]
        m = Statistics.mean(bin_data[!, :steer_cmd])
        s = Statistics.std(bin_data[!, :steer_cmd])
        inliers = filter(row -> abs(row.steer_cmd - m) < 1.5*s, bin_data)
        if nrow(inliers) > 0
          sampled_bin_data = inliers[sample(1:nrow(inliers), min(sample_size, nrow(inliers))), :]
          push!(bin_dfs[Threads.threadid()], sampled_bin_data)
        else
          sampled_bin_data = bin_data[sample(1:nrow(bin_data), min(sample_size, nrow(bin_data))), :]
          push!(bin_dfs[Threads.threadid()], sampled_bin_data)
        end
        
        next!(prog)
    end
    flattened_sampled_bin_data = SplitApplyCombine.flatten(bin_dfs)
    data = vcat(flattened_sampled_bin_data...)
    # data = balanced_data

    println(f"Min count bin has {minimum([size(i,1) for i in flattened_sampled_bin_data])} samples")

    CSV.write(joinpath(dirname(infile), replace(infile, ".csv" => "_balanced.csv")), data)
  end

  return data
end

function train_model(model_path::String, use_existing_model::Bool, data::DataFrame)::NamedTuple{(:model, :input_mean, :input_std, :X_train, :y_train, :X_test, :y_test), Tuple{Flux.Chain, Matrix{Float64}, Matrix{Float64}, Matrix{Float64}, Vector{Float64}, Matrix{Float64}, Vector{Float64}}}
  # split into train and test sets
  train, test = stratifiedobs(row->row[:combined_column], data, p = 0.8)

  # remove columns used only for binning
  select!(data, Not([:combined_column]))
  select!(data, Not([:v_ego_bins]))
  select!(data, Not([:lateral_accel_bins]))
  select!(data, Not([:lateral_accel_rate_no_roll_bins]))
  select!(data, Not([:roll_bins]))
  
  # split into independent an dependent variables
  X = Matrix(data[:, 1:end-1])  # Assuming the last column is the dependent_var
  y = data[:, :steer_cmd];

  # Calculate the mean and standard deviation of the input features
  input_mean = mean(X, dims=1)
  input_std = std(X, dims=1)

  # Create a copy of the DataFrames with the signs of steer_cmd, lateral_accel, lateral_jerk, and roll reversed to make the data symmetric
  old_size = size(train, 1)
  println("Training data before copying symmmetric data: $old_size")
  data_sym = copy(train)
  data_sym[!, :steer_cmd] = -1 .* data_sym[!, :steer_cmd]
  data_sym[!, :lateral_accel] = -1 .* data_sym[!, :lateral_accel]
  data_sym[!, :lateral_accel_rate_no_roll] = -1 .* data_sym[!, :lateral_accel_rate_no_roll]
  data_sym[!, :roll] = -1 .* data_sym[!, :roll]
  train = vcat(train, data_sym)
  println("Training data after copying symmmetric data: $(size(train,1))")

  old_size = size(test, 1)
  println("Test data before copying symmmetric data: $old_size")
  data_sym = copy(test)
  data_sym[!, :steer_cmd] = -1 .* data_sym[!, :steer_cmd]
  data_sym[!, :lateral_accel] = -1 .* data_sym[!, :lateral_accel]
  data_sym[!, :lateral_accel_rate_no_roll] = -1 .* data_sym[!, :lateral_accel_rate_no_roll]
  data_sym[!, :roll] = -1 .* data_sym[!, :roll]
  test = vcat(test, data_sym)
  println("Test data after copying symmmetric data: $(size(test,1))")

  # normalize the data
  X_train = Matrix(train[:, 1:end-1])
  X_train = (X_train .- input_mean) ./ input_std;
  y_train = train[:, "steer_cmd"];
  X_test = Matrix(test[:, 1:end-1])
  X_test = (X_test .- input_mean) ./ input_std;
  y_test = test[:, "steer_cmd"];

  input_dim = size(X_train, 2)

  # Define the model
  model = Chain(
      Dense(input_dim, 16, sigmoid),
      Dense(16, 8, sigmoid),
      Dense(8, 4),
      Dense(4, 1)
  )

  # Define L2 regularization strength
  l2_reg_strength = 0.002
  println("L2 regularization strength: $l2_reg_strength")
  # Define the loss function with L2 regularization
  loss(x, y) = Flux.mse(model(x), y) + l2_reg_strength * sum(x->x^2, Flux.params(model))

  # pick an optimizer
  # opt = Flux.ADAM(0.001)
  # opt = Flux.Nesterov()
  opt = Flux.AdaGrad()

  # train the model (with batches of shuffled data)
  tol = log10(size(X_train, 1)) > 7 ? 1e-4 : 7e-5
  Δtol = log10(size(X_train, 1)) > 7 ? 1e-4 : 7e-5
  logstep = log10(size(X_train, 1)) > 7 ? 3 : 10
  logstepgrowth = 1
  logstepfloat = Float64(logstep)
  logstepbig = 10
  Δloss = Inf
  ΔΔloss = Inf
  Δloss_last = 0.0
  loss_last = Inf
  loss_cur = 0.0
  epoch = 1
  epoch_max = log10(size(X_train, 1)) > 6 ? 150 : 300
  epoch_min = 25
  batch_size = 500
  train_data_loader = DataLoader((X_train', y_train), batchsize=batch_size, shuffle=true)

  # X_train = Array{Float64}(X_train)
  # y_train = Array{Float64}(y_train)
  # X_test = Array{Float64}(X_test)
  # y_test = Array{Float64}(y_test)

  println(size(X_train))
  println(size(y_train))

  if use_existing_model
      old_model = "$model_path.bson" #"/Users/haiiro/NoSync/voltlat.bson"
      println("Loading old model, $old_model")
      @load old_model model
  else
    while epoch < epoch_min || ((abs(Δloss) > tol || abs(ΔΔloss) > Δtol) && epoch < epoch_max)
        for (X_batch, y_batch) in train_data_loader
            println("Size of y_batch: $(size(y_batch))")
            train!(loss, params(model), [(X_batch, y_batch)], opt)
        end

        # train!(loss, params(model), [(X_train', y_train)], opt)
        
        if (epoch % logstep == 0 || epoch == 1)
            loss_cur = loss(X_train', y_train)
            Δloss = loss_cur - loss_last
            ΔΔloss = Δloss - Δloss_last
            loss_last = loss_cur
            Δloss_last = Δloss
            if abs(Δloss) > tol || abs(Δloss_last) > Δtol
                c1 = abs(Δloss) > tol ? ">" : "≤"
                c2 = abs(ΔΔloss) > Δtol ? ">" : "≤"
                println(f"Epoch: {epoch:3d} (of {epoch_max}), Loss: {loss_cur:.6f}, ΔLoss: {Δloss:.6f} {c1} {tol:.6G}, ΔΔLoss: {ΔΔloss:.6f} {c2} {Δtol:.6G},  Test loss: {loss(X_test', y_test):.6f}")
            end
            logstepfloat *= logstepgrowth
            logstep = round(Int, logstepfloat)
        end
        epoch += 1
    end
    # save the model for easy loading back into Flux.jl
    @save "$model_path.bson" model # "/Users/haiiro/NoSync/voltlat.bson" model
    println("Finished after $epoch epochs, Loss: $loss_cur, ΔLoss: $Δloss, Test loss: $(loss(X_test', y_test))")

  end

  # save model to json for Python import
  function export_model_params_to_json(model::Chain, input_mean::Matrix{Float64}, input_std::Matrix{Float64}, filename::String)
      W, b = params(model.layers[1])
      input_size = size(W, 2)
      output_size = size(params(model.layers[end])[1], 1)
      params_dict = Dict{String, Any}("input_size" => input_size, "output_size" => output_size, "layers" => [], "input_mean" => input_mean, "input_std" => input_std)

      for (idx, layer) in enumerate(model.layers)
          if isa(layer, Dense)
              W, b = params(layer)
              params_dict["layers"] = push!(params_dict["layers"], Dict(
                  "dense_$(idx)_W" => Array(W),
                  "dense_$(idx)_b" => Array(b),
                  "activation" => string(layer.σ)
              ))
          end
      end

      open(filename, "w") do f
          write(f, JSON.json(params_dict))
      end
  end

  export_model_params_to_json(model, Matrix{Float64}(input_mean), Matrix{Float64}(input_std), "$model_path.json")


  return (model=model, input_mean=input_mean, input_std=input_std, X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test)

end

function test_plot_model(model::Flux.Chain, plot_path::String, X_train::Matrix{Float64}, y_train::Vector{Float64}, X_test::Matrix{Float64}, y_test::Vector{Float64}, input_mean::Matrix{Float64}, input_std::Matrix{Float64})

    # Define L2 regularization strength
  l2_reg_strength = 0.002
  println("L2 regularization strength: $l2_reg_strength")
  # Define the loss function with L2 regularization
  loss(x, y) = Flux.mse(model(x), y) + l2_reg_strength * sum(x->x^2, Flux.params(model))

  # Evaluate the model on the test set and create plots
  test_loss = loss(X_test', y_test)
  println("Test loss (MSE): ", test_loss)


  function feedforward_function(input_data)
    # Scale the input data using the stored mean and standard deviation values
    input_data_scaled = (input_data .- input_mean) ./ input_std
    steer_command = model(input_data_scaled')
    return steer_command[1]
  end

  # Example input (v_ego	lateral_accel	lateral_jerk g_lat_accel)
  example_input = [25 0.5 0.1 0.2]
  steer_command = feedforward_function(example_input)
  println("Steer command: ", steer_command)



  # Create a function to filter the dataset based on speed
  function filter_data_by_speed(Xi, yi, speed, tolerance; no_jerk=false, no_roll=false)
    indices = findall(abs.(Xi[:, 1] .- speed) .< tolerance)
    X, y = Xi[indices, :], yi[indices]
    if no_jerk
      indices = findall(abs.(X[:, 3]) .< 0.2)
      X, y = X[indices, :], y[indices]
    end
    if no_roll
      indices = findall(abs.(X[:, 4]) .< 0.2)
      X, y = X[indices, :], y[indices]
    end
    return X, y
  end

  X_train_rescaled = X_train .* input_std .+ input_mean
  X_test_rescaled = X_test .* input_std .+ input_mean

  plot_scatter_step = round(Int, max(1, size(X_train, 1) / 10000))

  # Iterate over the speed range and create a plot for each speed
  # first w.r.t. lateral jerk
  speed_step = 10
  speed_range = 0:speed_step:30
  lateral_acceleration_range = range(-4.0, 4.0, length=100)

  plot_col_num = 1
  p = plot(layout = (size(collect(speed_range), 1), 2), legend=:bottomright, size=(1200, 1800), margin=10mm)

  for (si, speed) in enumerate(speed_range)
    # Plot the training data
    X_train_filtered, y_train_filtered = filter_data_by_speed(X_train_rescaled, y_train, speed, speed_step/2, no_roll=true)
    scatter!(p[si,plot_col_num], X_train_filtered[1:plot_scatter_step:end, 2], y_train_filtered[1:plot_scatter_step:end], label="Training Data", markersize=2, markercolor=:blue, markeralpha=0.1, xlims=(-3.5, 3.5), ylims=(-1.4,1.4))

    # Plot the test data
    X_test_filtered, y_test_filtered = filter_data_by_speed(X_test_rescaled, y_test, speed, speed_step/2, no_roll=true)
    scatter!(p[si,plot_col_num], X_test_filtered[1:plot_scatter_step:end, 2], y_test_filtered[1:plot_scatter_step:end], label="Test Data", markersize=2, markercolor=:red, markeralpha=0.3, xlims=(-3.5, 3.5), ylims=(-1.4,1.4))

    # Plot the model output
    for lj in -1:0.5:1
      x_model = []
      y_model = []
      for la in lateral_acceleration_range
          # (v_ego	lateral_accel	lateral_accel_rate_no_roll g_lat_accel)
          input_data = [speed la lj 0.0]
          steer_command = feedforward_function(input_data)
          push!(x_model, la)
          push!(y_model, steer_command)
      end
      plot!(p, xticks=true, yticks=true)
      plot!(p[si,plot_col_num], x_model, y_model, label="j_lat = $lj)", linewidth=4, xlims=(-3.5, 3.5), ylims=(-1.4,1.4))
    end


    # Configure the plot's appearance
    if si == 1
      title!(p[1,plot_col_num], f"Steer vs. a_lat at {speed*2.24:.2G}-{(speed+speed_step)*2.24:.2G} mph\nwith lateral jerk")
    else
      title!(p[si,plot_col_num], f"{speed*2.24:.2G}-{(speed+speed_step)*2.24:.2G} mph")
    end
    ylabel!(p[si,plot_col_num], "Steer Command")
  end
  xlabel!(p[size(collect(speed_range),1),plot_col_num], "a_lat (m/s²)")

  # now w.r.t. lateral gravitational acceleration

  # Iterate over the speed range and create a plot for each speed
  speed_step = 10
  speed_range = 0:speed_step:30
  lateral_acceleration_range = range(-4.0, 4.0, length=100)

  plot_col_num += 1

  for (si, speed) in enumerate(speed_range)

    # Plot the training data
    X_train_filtered, y_train_filtered = filter_data_by_speed(X_train_rescaled, y_train, speed, speed_step/2, no_jerk=true)
    scatter!(p[si,plot_col_num], X_train_filtered[1:plot_scatter_step:end, 2], y_train_filtered[1:plot_scatter_step:end], label="Training Data", markersize=2, markercolor=:blue, markeralpha=0.1, xlims=(-3.5, 3.5), ylims=(-1.4,1.4))

    # Plot the test data
    X_test_filtered, y_test_filtered = filter_data_by_speed(X_test_rescaled, y_test, speed, speed_step/2, no_jerk=true)
    scatter!(p[si,plot_col_num], X_test_filtered[1:plot_scatter_step:end, 2], y_test_filtered[1:plot_scatter_step:end], label="Test Data", markersize=2, markercolor=:red, markeralpha=0.3, xlims=(-3.5, 3.5), ylims=(-1.4,1.4))

    # Plot the model output
    for gla in -1:0.5:1
      x_model = []
      y_model = []
      for la in lateral_acceleration_range
          # (v_ego	lateral_accel	lateral_accel_rate_no_roll g_lat_accel)
          input_data = [speed la 0.0 gla]
          steer_command = feedforward_function(input_data)
          push!(x_model, la)
          push!(y_model, steer_command)
      end
      plot!(p[si,plot_col_num], x_model, y_model, label="a_lat_g = $gla)", linewidth=4, xlims=(-3.5, 3.5), ylims=(-1.4,1.4))
    end

    # Configure the plot's appearance
    if si == 1
      title!(p[1,plot_col_num], f"Steer vs. a_lat at {speed*2.24:.2G}-{(speed+speed_step)*2.24:.2G} mph\nwith lateral gravitational accel")
    else
      title!(p[si,plot_col_num], f"{speed*2.24:.2G}-{(speed+speed_step)*2.24:.2G} mph")
    end
  end
  xlabel!(p[size(collect(speed_range),1),plot_col_num], "a_lat (m/s²)")
  # Display the plot
  savefig(p, plot_path)
  display(p)
end

function main()
  
  data = load_data("/Users/haiiro/NoSync/voltlat_med.csv", true)

  model, input_mean, input_std, X_train, y_train, X_test, y_test = train_model("/Users/haiiro/NoSync/voltlat.bson", false, data)
  
  test_plot_model(model, "/Users/haiiro/NoSync/voltlat.png", X_train, y_train, X_test, y_test, input_mean, input_std)
  
end


main()


