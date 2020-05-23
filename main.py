import utils.scanner as sc
import utils.transformer as tr
import utils.printer as pr
import utils.data_processor as dp
import regr.regressors as rp

if __name__ == '__main__':
    """
    read data
    """
    cars = sc.read_csv()
    print("\n\nDATA READ:")
    pr.show_head(cars)

    """
    prepare data
    """
    cars_without_q_marks = tr.remove_q_marks(cars)
    cars_numerized = tr.numerize(cars_without_q_marks)

    print("\n\nNORMALIZED LOSS:")
    pr.show_norm_loss(cars_numerized)

    cars_dropna = tr.drop_nan_rows(cars_numerized)

    print("\n\nDATA NORMALIZED:")
    """
    normalize
    """
    normalized_cars = tr.normalize_cars(cars_dropna)
    pr.show_head(normalized_cars)

    """
    ask user if he need to see all predicted
    """
    printPredicted = sc.ask_if_should_be_printed_all_predicted()

    """
    calculate
    """
    for REGRESSOR in rp.REGRESSORS:
        print("-" * 100)
        print("\n\n---PROCESSING " + str(REGRESSOR.__class__.__name__) + " METHOD---")
        dp.calc_resRMSE_for_hp_and_width(normalized_cars, REGRESSOR, printPredicted)




