function main()
% MAIN Entry point for a generic EnVar optimization loop.
% This version is problem-agnostic: it only requires a forward model handle
% that maps a control vector to an observation and optional constraint
% function handles. The bundled default configuration demonstrates the
% workflow on a simple convex function f(x) = x(1)^2 + x(2)^2.

    cfg = default_config();
    [state, history] = initialize_run(cfg);

    for iter = 1:cfg.maxIterations
        [ensemble, deviations] = build_ensemble(cfg, state.control);
        writematrix(ensemble', "a_r_ensemble.txt", 'Delimiter', ' ');

        observations = evaluate_ensemble(cfg, ensemble);

        bounds = step_bounds(deviations, cfg.ensembleSize);
        objective = @(w) Cost(state.observation, observations - state.observation, w, state.control, deviations, cfg.penaltyWeight);
        constraints = cfg.constraintFcn;
        options = optimset('GradConstr', 'off', 'GradObj', 'off', 'MaxFunEvals', 100000);

        [weights, ~] = fmincon(objective, zeros(cfg.ensembleSize, 1), [], [], [], [], bounds.lb, bounds.ub, constraints, options);
        max(abs(deviations * weights)); %#ok<NOPRT> print maximum step size

        state = update_control(cfg, state, deviations, weights);
        history = record_iteration(cfg, history, state, iter);
    end
end

% -------------------------------------------------------------------------
function cfg = default_config()
% DEFAULT_CONFIG Encapsulates all tunable parameters for the EnVar loop.
% Users can swap out the demo forward model for any deterministic or
% stochastic simulator by editing cfg.forwardModel and
% cfg.observationReducer.

    cfg.ensembleSize = 10;
    cfg.penaltyWeight = 1;
    cfg.maxIterations = 15;

    % Forward model: maps a control vector to an observation.
    cfg.forwardModel = @(x) demo_convex_model(x);

    % Observation reducer: aggregates raw model outputs to a scalar (or
    % vector) observation used by the cost function.
    cfg.observationReducer = @(y) y; % identity for scalar outputs

    % Constraint function: accepts weight vector w and returns [C, Ceq].
    % Leave empty to disable constraints or replace with @demo_constraints
    % to enforce a radius-limited step.
    cfg.constraintFcn = [];

    % Initialization
    cfg.initialControlFile = 'a_Initial.txt';
    cfg.initialControlFallback = [-0.25; 0.75];
end

% -------------------------------------------------------------------------
function [state, history] = initialize_run(cfg)
    if isfile(cfg.initialControlFile)
        state.control = importdata(cfg.initialControlFile);
    else
        state.control = cfg.initialControlFallback;
    end
    state.initialNorm = norm(state.control)^2;
    state.controlHistory = state.control;

    rawObservation = cfg.forwardModel(state.control);
    state.observation = cfg.observationReducer(rawObservation);

    penaltyVector = (1:numel(state.control))' .* state.control;
    history.costFunction = state.observation;
    history.objective = 0.5 * norm(state.observation)^2 + 0.5 * cfg.penaltyWeight * norm(penaltyVector)^2;
end

% -------------------------------------------------------------------------
function [ensemble, deviations] = build_ensemble(cfg, controlVector)
    ensemble = Ensemble(cfg.ensembleSize, controlVector);
    deviations = ensemble - controlVector;
end

% -------------------------------------------------------------------------
function observations = evaluate_ensemble(cfg, ensemble)
% EVALUATE_ENSEMBLE Runs the forward model for each ensemble member and
% reduces the output to the observation space expected by the cost
% function.

    observations = zeros(cfg.ensembleSize, 1);
    for k = 1:cfg.ensembleSize
        raw = cfg.forwardModel(ensemble(:, k));
        observations(k) = cfg.observationReducer(raw);
    end
end

% -------------------------------------------------------------------------
function bounds = step_bounds(deviations, ensembleSize)
    aaa = max(abs(deviations));
    bbb = 10.^floor(log10(aaa));
    bbb = ((1 ./ bbb))' * ensembleSize;
    bounds.lb = -ones(ensembleSize, 1) .* bbb / ensembleSize^2;
    bounds.ub = ones(ensembleSize, 1) .* bbb / ensembleSize^2;
end

% -------------------------------------------------------------------------
function state = update_control(cfg, state, deviations, weights)
    state.control = state.control + deviations * weights;
    state.controlHistory(:, end + 1) = state.control;
    writematrix(state.control, "a_e.txt", 'Delimiter', ' ');

    rawObservation = cfg.forwardModel(state.control);
    state.observation = cfg.observationReducer(rawObservation);
end

% -------------------------------------------------------------------------
function history = record_iteration(cfg, history, state, iteration)
    writematrix([iteration, state.observation], "Cost_Iter.txt", 'Delimiter', ' ');

    penaltyVector = (1:numel(state.control))' .* state.control;
    iterationObjective = 0.5 * norm(state.observation)^2 + 0.5 * cfg.penaltyWeight * norm(penaltyVector)^2;

    history.costFunction(1, iteration + 1) = state.observation;
    history.objective(iteration + 1, 1) = iterationObjective;

    writematrix(state.controlHistory, "Evolution_a_e.txt", 'Delimiter', ' ');
    writematrix(history.costFunction', "Cost_Function.txt", 'Delimiter', ' ');
    writematrix(history.objective, "J_Obj.txt", 'Delimiter', ' ');
end

% -------------------------------------------------------------------------
function y = demo_convex_model(x)
% DEMO_CONVEX_MODEL Simple convex objective used as the default forward
% model. It demonstrates the EnVar architecture without any CFD or PDE
% dependencies.

    y = sum(x.^2);
end

% -------------------------------------------------------------------------
function [C, Ceq] = demo_constraints(w)
% DEMO_CONSTRAINTS Optional example constraint: limit the step norm to 0.5.
% Replace cfg.constraintFcn with @demo_constraints to activate.

    maxRadius = 0.5;
    C = norm(w) - maxRadius;
    Ceq = [];
end
