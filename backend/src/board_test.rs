use super::board::*;
use std::cmp::Ordering;

#[test]
fn test_board_create() {
    let board = Board::new(String::from("121211"), 1, 6);
    assert_eq!(board.width, 1);
    assert_eq!(board.height, 6);
    assert_eq!(board.to_string(), "121211");
}

#[test]
fn test_board_copy() {
    let board = Board::new(String::from("121211"), 1, 6);
    let mut copy = board.clone();
    copy.place(0, 0, 0);
    assert_eq!(board.get(0, 0), Some(1));
    assert_eq!(copy.get(0, 0), Some(0));
}

#[test]
fn test_line() {
    let mut line = Line::new(5, 0, 0);
    assert_eq!(line.is_non_refutable(), false);

    line = Line::new(5, 0, 1);
    assert_eq!(line.is_non_refutable(), false);

    line = Line::new(5, 1, 0);
    assert_eq!(line.is_non_refutable(), false);
}

#[test]
fn test_line_compare() {
    let line1 = Line::new(3, 0, 1);
    let line2 = Line::new(4, 0, 1);
    assert_eq!(line1.cmp(&line2), Ordering::Less);

    let line3 = Line::new(3, 1, 2);
    assert_eq!(line1.cmp(&line3), Ordering::Less);

    let line4 = Line::new(3, 0, 1);
    assert_eq!(line1.cmp(&line4), Ordering::Equal);

    let line5 = Line::new(5, 0, 0);
    assert_eq!(line1.cmp(&line5), Ordering::Less);
}

#[test]
fn test_board_from_string() {
    let board = Board::from(String::from("1212112121121211212112121"));
    assert_eq!(board.width, 5);
    assert_eq!(board.height, 5);
}

#[test]
#[should_panic(expected = "Invalid input string size")]
fn test_board_from_string_panic() {
    let _ = Board::from(String::from("12123"));
}

#[test]
#[should_panic(expected = "Invalid board size with 2*2 <> 2")]
fn test_board_size_validation() {
    Board::new(String::from("12"), 2, 2);
}

#[test]
fn test_board_elements() {
    let board = Board::new(String::from("000112000112"), 6, 2);
    assert_eq!(board.cells[0][0], 0);
    assert_eq!(board.cells[0][1], 0);
    assert_eq!(board.cells[1][4], 1);
    assert_eq!(board.cells[1][5], 2);
}

#[test]
fn test_board_check_winner() {
    let mut board = Board::new(String::from("1111100000"), 5, 2);
    assert_eq!(board.any_winner(), Some(1));

    board = Board::new(String::from("1111000000"), 5, 2);
    assert_eq!(board.any_winner(), None);

    board = Board::new(String::from("1111022222"), 5, 2);
    assert_eq!(board.any_winner(), Some(2));

    board = Board::new(String::from("111101111011110"), 5, 3);
    assert_eq!(board.any_winner(), None);

    board = Board::new(String::from("1111011110111101111010000"), 5, 5);
    assert_eq!(board.any_winner(), Some(1));

    board = Board::new(String::from("1111011110111100111010000"), 5, 5);
    assert_eq!(board.any_winner(), None);

    board = Board::new(String::from("10000 01000 00100 00010 00001"), 5, 5);
    assert_eq!(board.any_winner(), Some(1));

    board = Board::new(String::from("10000 01000 00000 00010 00001"), 5, 5);
    assert_eq!(board.any_winner(), None);

    board = Board::new(String::from("22220 10000 01000 00100 00010 00001"), 5, 6);
    assert_eq!(board.any_winner(), Some(1));

    board = Board::new(String::from("22220 00001 00010 00100 01000 10000"), 5, 6);
    assert_eq!(board.any_winner(), Some(1));

    board = Board::new(String::from("22222 00001 00010 00100 01000 10000"), 5, 6);
    assert_eq!(board.any_winner(), Some(2));

    board = Board::new(
        String::from(
            "
        10000
        10001
        01021
        00001
        00001
        00001",
        ),
        5,
        6,
    );

    let line = board.connect_direction(1, 1, 4, 1, 0, true);
    assert_eq!(line.count, 5);
    assert_eq!(board.any_winner(), Some(1));
}

#[test]
fn test_board_score() {
    let mut board = Board::new(String::from("1111020000"), 5, 2);
    assert_eq!(board.eval_pos(1, 0, 0), 1050);
    assert_eq!(board.eval_pos(2, 1, 0), 0);

    board = Board::new(String::from("1111111111"), 5, 2);
    assert_eq!(board.eval_all(2), 0);
    assert_eq!(board.eval_all(1), 1000000);

    board = Board::new(String::from("10000 01000 00100"), 5, 3);
    assert_eq!(board.eval_all(1), 0);

    board = Board::new(
        String::from(
            "
        10000
        01000
        00100
        00000",
        ),
        5,
        4,
    );
    assert_eq!(board.eval_all(1), 0);

    board = Board::new(
        String::from(
            "
        10000
        01000
        00100
        00000
        00000",
        ),
        5,
        5,
    );
    assert_eq!(board.eval_all(1), 90);

    board = Board::new(
        String::from(
            "
        10000
        01100
        00100
        00000",
        ),
        5,
        4,
    );
    assert_eq!(board.eval_all(1), 50);

    board = Board::new(
        String::from(
            "
            000000
            011100
            011100
            011100
            000000
            000000",
        ),
        6,
        6,
    );
    assert_eq!(board.eval_all(1), 27750);

    board = Board::new(
        String::from(
            "
        101100
        000000",
        ),
        6,
        2,
    );
    assert_eq!(board.eval_all(1), 75);

    board = Board::new(
        String::from(
            "
        1011100
        0000000",
        ),
        7,
        2,
    );
    assert_eq!(board.eval_all(1), 4180);

    board = Board::new(
        String::from(
            "
        0000000
        0001000
        0001000
        0022000
        0000000
        ",
        ),
        7,
        5,
    );
    assert_eq!(board.eval_all(1), 0);

    board = Board::new(
        String::from(
            "
        0000000
        0001000
        0001000
        0111000
        0000000
        ",
        ),
        7,
        5,
    );
    assert_eq!(board.eval_pos(1, 3, 3), 2300);

    board = Board::new(
        String::from(
            "
        0000000
        0001000
        0001000
        0020000
        0000000
        ",
        ),
        7,
        5,
    );
    assert_eq!(board.eval_all(1), 50);

    board = Board::new(
        String::from(
            "
        0000000
        0001000
        0001000
        0020000
        0000000
        0000000
        ",
        ),
        7,
        6,
    );
    assert_eq!(board.eval_all(1), 50);

    board = Board::new(
        String::from(
            "
        0000000
        0000000
        0001000
        0001000
        0001200
        0002000
        0000000
        ",
        ),
        7,
        7,
    );
    assert_eq!(board.eval_pos(1, 2, 3), 30);

    board = Board::new(
        String::from(
            "
        0000000
        0000000
        0000000
        0001100
        0001200
        0002000
        0000000
        ",
        ),
        7,
        7,
    );
    assert_eq!(board.eval_pos(1, 3, 4), 50);

    board = Board::new(
        String::from(
            "
        0000000
        0001000
        0001000
        0001000
        0001000
        0002000
        0002000
        ",
        ),
        7,
        7,
    );
    assert_eq!(board.eval_pos(1, 1, 3), 1050);
}

#[test]
fn test_one_direction_corner_case() {
    let mut board = Board::new(
        String::from(
            "
        0000000
        0000100
        0000000
        0000000
        0000000
        0000000
        ",
        ),
        7,
        6,
    );

    let line = board.connect_direction(1, 1, 4, 1, 1, true);
    assert_eq!(line, Line::new(1, 0, 0));
    let line = board.connect_direction(1, 1, 4, 1, 1, false);
    assert_eq!(line, Line::new(1, 1, 0));

    board = Board::new(
        String::from(
            "
        00000000
        00000000
        00000100
        00000000
        00000000
        00000000
        ",
        ),
        8,
        6,
    );

    let line = board.connect_direction(1, 2, 5, 1, 1, true);
    assert_eq!(line, Line::new(1, 0, 2));
    let line = board.connect_direction(1, 2, 5, 1, 1, false);
    assert_eq!(line, Line::new(1, 1, 2));

    board = Board::new(
        String::from(
            "
        00000000
        00000000
        00000100
        00000010
        00000000
        00000000
        ",
        ),
        8,
        6,
    );

    let line = board.connect_direction(1, 2, 5, 1, 1, true);
    assert_eq!(line, Line::new(2, 0, 2));

    board = Board::new(
        String::from(
            "
        00000000
        00000000
        00000100
        00000010
        00000000
        00000000
        ",
        ),
        8,
        6,
    );

    let line = board.connect_direction(1, 2, 5, 1, 1, true);
    assert_eq!(line, Line::new(2, 0, 2));

    board = Board::new(
        String::from(
            "
        00000000
        00000000
        00000100
        00000010
        00000002
        00000000
        ",
        ),
        8,
        6,
    );

    let line = board.connect_direction(1, 2, 5, 1, 1, true);
    assert_eq!(line, Line::new(2, 0, 0));

    board = Board::new(
        String::from(
            "
        00000000
        00000000
        00001000
        00000100
        00000010
        00000002
        ",
        ),
        8,
        6,
    );

    let line = board.connect_direction(1, 2, 4, 1, 1, true);
    assert_eq!(line, Line::new(3, 0, 1));

    let line = board.connect_direction(1, 3, 5, 1, 1, true);
    assert_eq!(line, Line::new(3, 0, 1));

    board = Board::new(
        String::from(
            "
        00000000
        00000000
        00201020
        00000000
        00000000
        00000000
        ",
        ),
        8,
        6,
    );

    let line = board.connect_direction(1, 2, 4, 0, 1, true);
    assert_eq!(line, Line::new(1, 0, 0));
}
