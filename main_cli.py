import argparse
from mvp.experiment import Experiment
from mvp import rest_server

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    sub_parsers = parser.add_subparsers(dest='action')

    experiment_parser = sub_parsers.add_parser('run_exp', description='Run Experiment')
    experiment_parser.add_argument('--n_lag_days', default=2, type=int, required=False)
    experiment_parser.add_argument('--cv_splits', default=3, type=int, required=False)
    # todo: future version allow to specify model arguments

    server_parser = sub_parsers.add_parser('start_rest', description='Start REST Server')
    server_parser.add_argument('--n_lag_days', default=2, type=int, required=False)
    server_parser.add_argument('--port', default=5000, type=int, required=False)
    server_parser.add_argument('--model_path', type=str, required=True)

    args = parser.parse_args()
    print(args)

    if args.action == 'run_exp':
        experiment = Experiment(cv_spilits=args.cv_splits, n_lag_days=args.n_lag_days)
        experiment.run()

    elif args.action == 'start_rest':
        rest_server.model_path = args.model_path
        rest_server.app.run(host='0.0.0.0', port=args.port)

    else:
        raise ValueError('unknown action')
