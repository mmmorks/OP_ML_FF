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

function main()
  # Load the data into a DataFrame (replace 'your_data.csv' with your file)
  infile = "/Users/haiiro/NoSync/voltlat_large.csv"
  if use_existing_data
    infile = "/Users/haiiro/NoSync/balanced_data.csv"
    println("Using existing data")
  end
  data = CSV.read(infile, DataFrame)
  if !use_existing_data
    data = data[completecases(data), :]

    println(f"Loaded {nrow(data)} rows")

    mm_v_ego = [2, 40]
    mm_steer_cmd = [-2.0, 2.0]
    mm_lateral_accel = [-4.45, 4.45]
    mm_lateral_accel_rate_no_roll = [-2.95, 2.95]
    mm_roll = [-2.95, 2.95]

    nbins = 80
    sample_size = 3
    step_v_ego = (mm_v_ego[2] - mm_v_ego[1]) / nbins
    step_steer_cmd = (mm_steer_cmd[2] - mm_steer_cmd[1]) / nbins
    step_lateral_accel = (mm_lateral_accel[2] - mm_lateral_accel[1]) / nbins
    step_lateral_accel_rate_no_roll = (mm_lateral_accel_rate_no_roll[2] - mm_lateral_accel_rate_no_roll[1]) / nbins
    step_roll = (mm_roll[2] - mm_roll[1]) / nbins

    data = filter(row -> mm_v_ego[1] < row.v_ego < mm_v_ego[2], data)
    data = filter(row -> abs(row.steer_cmd) <= mm_steer_cmd[2], data)
    data = filter(row -> abs(row.lateral_accel) < mm_lateral_accel[2], data)
    data = filter(row -> abs(row.lateral_accel_rate_no_roll) < mm_lateral_accel_rate_no_roll[2], data)
    data = filter(row -> abs(row.roll) < mm_roll[2], data)
    # data = filter(row -> abs(row.a_ego) < 2.95, data)
    select!(data, Not([:a_ego]))
    # data = DataFrame(vcat([data, df_reversed]))
    println(f"Filtered to {nrow(data)} rows")
    # println(data[1:10, :])


    # balanced_data = DataFrame()
    println(f"Calculating bins")
    data[!, :v_ego_bins] = cut(data[!, :v_ego], mm_v_ego[1]:step_v_ego:mm_v_ego[2])
    data[!, :lateral_accel_bins] = cut(data[!, :lateral_accel], mm_lateral_accel[1]:step_lateral_accel:mm_lateral_accel[2])
    data[!, :roll_bins] = cut(data[!, :roll], mm_roll[1]:step_roll:mm_roll[2])
    data[!, :lateral_accel_rate_no_roll_bins] = cut(data[!, :lateral_accel_rate_no_roll], mm_lateral_accel_rate_no_roll[1]:step_lateral_accel_rate_no_roll:mm_lateral_accel_rate_no_roll[2])


    data[!,:combined_column] = string.(data[!,:v_ego_bins], "_", data[!,:lateral_accel_bins])#, "_", data[!,:lateral_accel_rate_no_roll_bins], "_", data[!,:roll_bins])

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

    CSV.write(joinpath(dirname(infile),"balanced_data.csv"), data)
  end

  train, test = stratifiedobs(row->row[:combined_column], data, p = 0.8)

  select!(data, Not([:combined_column]))

  select!(data, Not([:v_ego_bins]))
  select!(data, Not([:lateral_accel_bins]))
  select!(data, Not([:lateral_accel_rate_no_roll_bins]))
  select!(data, Not([:roll_bins]))
  
  X = Matrix(data[:, 1:end-1])  # Assuming the last column is the dependent_var
  y = data[:, :steer_cmd];

  # Calculate the mean and standard deviation of the input features
  input_mean = mean(X, dims=1)
  input_std = std(X, dims=1)

  # select!(train, Not([:combined_column]))
  # select!(train, Not([:v_ego_bins]))
  # select!(train, Not([:lateral_accel_bins]))

  # select!(test, Not([:combined_column]))
  # select!(test, Not([:v_ego_bins]))
  # select!(test, Not([:lateral_accel_bins]))

  # Create a copy of the DataFrames with the signs of steer_cmd, lateral_accel, lateral_jerk, and roll reversed
  old_size = size(train, 1)
  println("Training data before copying symmmetric data: $old_size")
  data_sym = copy(train)
  data_sym[!, :steer_cmd] = -1 .* data_sym[!, :steer_cmd]
  data_sym[!, :lateral_accel] = -1 .* data_sym[!, :lateral_accel]
  data_sym[!, :lateral_accel_rate_no_roll] = -1 .* data_sym[!, :lateral_accel_rate_no_roll]
  data_sym[!, :roll] = -1 .* data_sym[!, :roll]
  # data_sym[!, :a_ego] = -1 .* data_sym[!, :a_ego]
  train = vcat(train, data_sym)
  # train[!, :a_ego] = abs.(train[!, :a_ego])
  println("Training data after copying symmmetric data: $(size(train,1))")
  # println(train[1:5, :])
  # println(train[1+old_size:5+old_size, :])

  old_size = size(test, 1)
  println("Test data before copying symmmetric data: $old_size")
  data_sym = copy(test)
  data_sym[!, :steer_cmd] = -1 .* data_sym[!, :steer_cmd]
  data_sym[!, :lateral_accel] = -1 .* data_sym[!, :lateral_accel]
  data_sym[!, :lateral_accel_rate_no_roll] = -1 .* data_sym[!, :lateral_accel_rate_no_roll]
  data_sym[!, :roll] = -1 .* data_sym[!, :roll]
  # data_sym[!, :a_ego] = -1 .* data_sym[!, :a_ego]
  test = vcat(test, data_sym)
  # test[!, :a_ego] = abs.(test[!, :a_ego])
  println("Test data after copying symmmetric data: $(size(test,1))")
  # println(test[1:5, :])
  # println(test[1+old_size:5+old_size, :])

  X_train = Matrix(train[:, 1:end-1])
  X_train = (X_train .- input_mean) ./ input_std;
  y_train = train[:, "steer_cmd"];
  X_test = Matrix(test[:, 1:end-1])
  X_test = (X_test .- input_mean) ./ input_std;
  y_test = test[:, "steer_cmd"];
  # train_indices, test_indices = splitobs(collect(1:size(X_scaled, 1)), at=0.8)

  # X_train = X[train_indices, :]
  # y_train = y[train_indices]
  # X_test = X[test_indices, :]
  # y_test = y[test_indices]

  size(X_train), size(y_train), size(X_test), size(y_test)

  input_dim = size(X_train, 2)

  model = Chain(
      Dense(input_dim, 16, sigmoid),
      Dense(16, 8, sigmoid),
      Dense(8, 4),
      Dense(4, 2),
      Dense(2, 1)
  )

  function weighted_mse(ŷ, y, x)
    weights = map(xi -> abs(xi[1]) > 0.01 ? 10.0 : 1.0, eachcol(x))
    return sum((ŷ - y) .^ 2 .* weights) / length(y)
  end

  function combined_weighted_loss(ŷ, y, x)
    mse_loss = weighted_mse(ŷ, y, x)
    wd_loss = weight_decay(params(model), λ)
    return mse_loss + wd_loss
  end

  # Define L2 regularization strength
  l2_reg_strength = 0.002
  println("L2 regularization strength: $l2_reg_strength")
  # Define the loss function with L2 regularization
  loss(x, y) = Flux.mse(model(x), y) + l2_reg_strength * sum(x->x^2, Flux.params(model))


  training_data = [(X_train', y_train)]


  # Use the ADAM optimizer with a learning rate of 0.001
  # opt = Flux.ADAM(0.001)
  # opt = Flux.Nesterov()
  opt = Flux.AdaGrad()

  tol = log10(size(X_train, 1)) > 7 ? 5e-5 : 8e-6
  logstep = log10(size(X_train, 1)) > 7 ? 3 : 10
  logstepgrowth = 1
  logstepfloat = Float32(logstep)
  logstepbig = 10
  Δloss = Inf
  loss_last = Inf
  loss_cur = 0.0
  epoch = 1
  epoch_max = log10(size(X_train, 1)) > 6 ? 150 : 5000
  epoch_min = 25
  batch_size = 100*3600
  # train_data_loader = DataLoader((X_train', y_train), batchsize=batch_size, shuffle=true)

  X_train = Array{Float32}(X_train)
  y_train = Array{Float32}(y_train)
  X_test = Array{Float32}(X_test)
  y_test = Array{Float32}(y_test)

  println(size(X_train))
  println(size(y_train))

  if use_existing_model
      old_model = "/Users/haiiro/NoSync/voltlat.bson"
      println("Loading old model, $old_model")
      @load old_model model
      loss(x, y) = mse(model(x), y')
  else

    num_batches = ceil(Int, size(X_train, 1) / batch_size)
    all_indices = randperm(size(X_train, 1))
    batch_indices = [all_indices[((i - 1) * batch_size + 1):min(i * batch_size, size(X_train, 1))] for i in 1:num_batches]

    # prog = ProgressThresh(tol, "Minimizing:")

    while epoch < epoch_min || (Δloss > tol && epoch < epoch_max)
        # train!(loss, params(model), training_data, opt)
        # for (X_batch, y_batch) in train_data_loader
        #     train!(loss, params(model), [(X_batch, y_batch)], opt)
        # end
        # Shuffle the indices before each epoch
        shuffle!(all_indices)
        batch_indices = [all_indices[((i - 1) * batch_size + 1):min(i * batch_size, size(X_train, 1))] for i in 1:num_batches]
        # Iterate over mini-batches in parallel
        @threads for i in 1:num_batches
            indices = batch_indices[i]
            X_batch = X_train[indices, :]'
            y_batch = y_train[indices]
            train!(loss, params(model), [(X_batch, y_batch)], opt)
        end
        
        if (epoch % logstep == 0 || epoch == 1)
            loss_cur = loss(X_train', y_train)
            Δloss = abs(loss_cur - loss_last)
            loss_last = loss_cur
            # ProgressMeter.update!(prog, Δloss)
            if Δloss > tol
                # if epoch % logstepbig != 0
                #     print("\e[2K") # clear whole line
                #     print("\e[1G") # move cursor to column 1
                # else
                #     print("\n")
                # end
                println(f"Epoch: {epoch:3d} (of {epoch_max}), Loss: {loss_cur:.6f}, ΔLoss: {Δloss:.6f} > {tol:.6G}, Test loss: {loss(X_test', y_test):.6f}")
            end
            logstepfloat *= logstepgrowth
            logstep = round(Int, logstepfloat)
        end
        epoch += 1
    end
    @save "/Users/haiiro/NoSync/voltlat.bson" model
    println("Finished after $epoch epochs, Loss: $loss_cur, ΔLoss: $Δloss, Test loss: $(loss(X_test', y_test))")
    # of = open("/Users/haiiro/NoSync/voltlat.scale.txt", "w")
    # write(of, f"mean = {input_mean}
    # std = {input_std}
    # function feedforward_function(input_data)
    #     # Scale the input data using the stored mean and standard deviation values
    #     input_data_scaled = (input_data .- input_mean) ./ input_std
    #     steer_command = model(input_data_scaled')
    #     return steer_command[1]
    # end
    # # Example input (v_ego  -lateral_accel	-lateral_accel_rate_no_roll g_lat_accel)")
    # close(of)

  end

  # save model to json for Python import
  function export_model_params_to_json(model::Chain, input_mean::Matrix{Float32}, input_std::Matrix{Float32}, filename::String)
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


  export_model_params_to_json(model, Matrix{Float32}(input_mean), Matrix{Float32}(input_std), "/Users/haiiro/NoSync/voltlat.json")

  test_loss = loss(X_test', y_test)
  println("Test loss (MSE): ", test_loss)


  function feedforward_function(input_data)
    # Scale the input data using the stored mean and standard deviation values
    input_data_scaled = (input_data .- input_mean) ./ input_std
    steer_command = model(input_data_scaled')
    return steer_command[1]
  end

  # Example input (v_ego	a_ego	lateral_accel	lateral_accel_rate_no_roll g_lat_accel)
  # example_input = [25 1 0.5 0.1 0.2]
  # Example input (v_ego	lateral_accel	lateral_accel_rate_no_roll g_lat_accel)
  example_input = [25 0.5 0.1 0.2]
  steer_command = feedforward_function(example_input)
  println("Steer command: ", steer_command)



  # Create a function to filter the dataset based on speed
  function filter_data_by_speed(X, y, speed, tolerance)
    indices = findall(abs.(X[:, 1] .- speed) .< tolerance)
    return X[indices, :], y[indices]
    # indices = findall(abs.(X[:, 2]) .< 0.1)
    # X, y = X[indices, :], y[indices]
    # indices = findall(abs.(X[:, 4]) .< 0.03)
    # X, y = X[indices, :], y[indices]
    
  end

  X_train_rescaled = X_train .* input_std .+ input_mean
  X_test_rescaled = X_test .* input_std .+ input_mean

  plot_scatter_step = round(Int, max(1, size(X_train, 1) / 10000))

  # Iterate over the speed range and create a plot for each speed
  speed_step = 10
  speed_range = 0:speed_step:30
  lateral_acceleration_range = range(-4.0, 4.0, length=100)

  plot_col_num = 1
  p = plot(layout = (size(collect(speed_range), 1), 2), legend=:bottomright, size=(1200, 1800), margin=10mm)

  for (si, speed) in enumerate(speed_range)
    # Plot the training data
    X_train_filtered, y_train_filtered = filter_data_by_speed(X_train_rescaled, y_train, speed, speed_step/2)
    scatter!(p[si,plot_col_num], X_train_filtered[1:plot_scatter_step:end, 2], y_train_filtered[1:plot_scatter_step:end], label="Training Data", markersize=2, markercolor=:blue, markeralpha=0.1, xlims=(-3.5, 3.5), ylims=(-1.2,1.2))

    # Plot the test data
    X_test_filtered, y_test_filtered = filter_data_by_speed(X_test_rescaled, y_test, speed, speed_step/2)
    scatter!(p[si,plot_col_num], X_test_filtered[1:plot_scatter_step:end, 2], y_test_filtered[1:plot_scatter_step:end], label="Test Data", markersize=2, markercolor=:red, markeralpha=0.3, xlims=(-3.5, 3.5), ylims=(-1.2,1.2))

    # Plot the model output
    for lj in -1:0.5:1
      x_model = []
      y_model = []
      for la in lateral_acceleration_range
          # (v_ego	a_ego	lateral_accel	lateral_accel_rate_no_roll g_lat_accel)
          input_data = [speed la lj 0.0]
          # input_data = [speed 0.0 la lj 0.0]
          steer_command = feedforward_function(input_data)
          push!(x_model, la)
          push!(y_model, steer_command)
      end
      plot!(p[si,plot_col_num], x_model, y_model, label="j_lat = $lj)", linewidth=4, xlims=(-3.5, 3.5), ylims=(-1.2,1.2))
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


  # # Iterate over the speed range and create a plot for each speed
  # speed_step = 10
  # speed_range = 0:speed_step:30
  # lateral_acceleration_range = range(-4.0, 4.0, length=100)

  # for (si, speed) in enumerate(speed_range)
  #   p = plot(legend=:bottomright)

  #   # Plot the training data
  #   X_train_filtered, y_train_filtered = filter_data_by_speed(X_train_rescaled, y_train, speed, speed_step/2)
  #   scatter!(p, X_train_filtered[1:plot_scatter_step:end, 3], y_train_filtered[1:plot_scatter_step:end], label="Training Data", markersize=2, markercolor=:blue, markeralpha=0.1, xlims=(-3.5, 3.5), ylims=(-1.2,1.2))

  #   # Plot the test data
  #   X_test_filtered, y_test_filtered = filter_data_by_speed(X_test_rescaled, y_test, speed, speed_step/2)
  #   scatter!(p, X_test_filtered[1:plot_scatter_step:end, 3], y_test_filtered[1:plot_scatter_step:end], label="Test Data", markersize=2, markercolor=:red, markeralpha=0.1, xlims=(-3.5, 3.5), ylims=(-1.2,1.2))

  #   # Plot the model output
  #   for aego in -3.0:1.5:3.0
  #     x_model = []
  #     y_model = []
  #     for la in lateral_acceleration_range
  #         # (v_ego	a_ego	lateral_accel	lateral_accel_rate_no_roll g_lat_accel)
  #         input_data = [speed aego la 0.0 0.0]
  #         steer_command = feedforward_function(input_data)
  #         push!(x_model, la)
  #         push!(y_model, steer_command)
  #     end
  #     plot!(p, x_model, y_model, label="Model (aego=$aego)", linewidth=4, xlims=(-3.5, 3.5), ylims=(-1.2,1.2))
  #   end


  #   # Configure the plot's appearance
  #   xlabel!(p, "a_lat (m/s²)")
  #   ylabel!(p, "Steer Command")
  #   title!(p, f"Steer vs. a_lat at {speed*2.24:.2G}-{(speed+speed_step)*2.24:.2G} mph")

  #   # Display the plot
  #   display(p)
  # end



  # Iterate over the speed range and create a plot for each speed
  speed_step = 10
  speed_range = 0:speed_step:30
  lateral_acceleration_range = range(-4.0, 4.0, length=100)

  plot_col_num += 1

  for (si, speed) in enumerate(speed_range)

    # Plot the training data
    X_train_filtered, y_train_filtered = filter_data_by_speed(X_train_rescaled, y_train, speed, speed_step/2)
    scatter!(p[si,plot_col_num], X_train_filtered[1:plot_scatter_step:end, 2], y_train_filtered[1:plot_scatter_step:end], label="Training Data", markersize=2, markercolor=:blue, markeralpha=0.1, xlims=(-3.5, 3.5), ylims=(-1.2,1.2))

    # Plot the test data
    X_test_filtered, y_test_filtered = filter_data_by_speed(X_test_rescaled, y_test, speed, speed_step/2)
    scatter!(p[si,plot_col_num], X_test_filtered[1:plot_scatter_step:end, 2], y_test_filtered[1:plot_scatter_step:end], label="Test Data", markersize=2, markercolor=:red, markeralpha=0.3, xlims=(-3.5, 3.5), ylims=(-1.2,1.2))

    # Plot the model output
    for gla in -1:0.5:1
      x_model = []
      y_model = []
      for la in lateral_acceleration_range
          # (v_ego	a_ego	lateral_accel	lateral_accel_rate_no_roll g_lat_accel)
          # input_data = [speed 0.0 la 0.0 gla]
          input_data = [speed la 0.0 gla]
          steer_command = feedforward_function(input_data)
          push!(x_model, la)
          push!(y_model, steer_command)
      end
      plot!(p[si,plot_col_num], x_model, y_model, label="a_lat_g = $gla)", linewidth=4, xlims=(-3.5, 3.5), ylims=(-1.2,1.2))
    end


    # Configure the plot's appearance
    if si == 1
      title!(p[1,plot_col_num], f"Steer vs. a_lat at {speed*2.24:.2G}-{(speed+speed_step)*2.24:.2G} mph\nwith lateral gravitational accel")
    else
      title!(p[si,plot_col_num], f"{speed*2.24:.2G}-{(speed+speed_step)*2.24:.2G} mph")
    end

    # ylabel!(p[si,plot_col_num], "Steer Command")
  end
  xlabel!(p[size(collect(speed_range),1),plot_col_num], "a_lat (m/s²)")
  # title!(p,"Steer Command vs Lateral Acceleration (Residuals) (MSE: $(round(test_loss, digits=4)))")
  # Display the plot
  savefig(p, "/Users/haiiro/NoSync/voltlat.png")
  display(p)
end


main()


