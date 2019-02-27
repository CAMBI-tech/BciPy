"""Reads log file for a session and converts feedback decisions to csv
document for analysis."""
import csv


def main(logfile: str, outfile: str):
    """Reads through the logfile and extracts the feedback into the outfile
    as a csv."""
    with open(outfile, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['response_decision', 'feedback_position'])
        with open(logfile) as infile:
            decision = None
            for line in infile:
                if "[Feedback] Response decision" in line:
                    decision = line.split()[-1]
                elif "[Feedback] Administering feedback" in line:
                    feedback_value = line.split()[-1]
                    if not decision:
                        raise Exception(
                            "Feedback value encountered without a matching decision"
                        )
                    writer.writerow([decision, feedback_value])
                    decision = None
                    feedback_value = None
    print(f"CSV written to {outfile}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-l', '--logfile', required=True, help='Path to log file.')
    parser.add_argument('--csv', default='feedback_decisions.csv')

    args = parser.parse_args()
    main(logfile=args.logfile, outfile=args.csv)
