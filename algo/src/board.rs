#![allow(dead_code)]

#[derive(Debug)]
pub struct Board {
    width: usize,
    height: usize,
    digits: Vec<Vec<u8>>,
}

#[derive(Debug)]
struct Line {
    count: usize,
    hole_count: i32,
    open_count: i32,
}

impl Line {
    pub fn new(count: usize, hole_count: i32, open_count: i32) -> Self {
        Line {
            count: count,
            hole_count: hole_count,
            open_count: open_count,
        }
    }

    pub fn is_non_refutable(&self) -> bool {
        (self.count >= 5 && self.hole_count == 0)
            || (self.count == 4 && self.hole_count == 0 && self.open_count == 2)
    }

    pub fn score(&self) -> u32 {
        match (self.count, self.hole_count, self.open_count) {
            (5, 0, _) => 10000,
            (4, 0, 2) => 9900,
            (5, 1, _) => 1000,
            (4, 0, 1) => 1500,
            (4, 1, 2) => 2000,
            (4, 1, 0) => 800,
            (3, 0, 2) => 1000,
            (3, 0, 1) => 500,
            (3, 1, 2) => 400,
            (3, 1, 0) => 200,
            (3, 0, 0) => 100,
            (2, 0, 2) => 80,
            (2, 0, 1) => 10,
            (_, _, _) => 0,
        }
    }
}

impl Board {
    pub fn new(input: String, width: usize, height: usize) -> Self {
        let rows: Vec<u8> = input
            .chars()
            .filter(|&e| e == '0' || e == '1' || e == '2')
            .map(|e| e as u8 - '0' as u8)
            .collect();
        if rows.len() != width * height {
            panic!(
                "Invalid board size with {}*{} <> {}",
                width,
                height,
                rows.len()
            );
        }
        if width < 5 && height < 5 {
            panic!("Width or height must larger than")
        }
        Self {
            width: width,
            height: height,
            digits: rows.chunks(width).map(|x| x.to_vec()).collect(),
        }
    }

    pub fn from(input: String) -> Self {
        let len = input.len();
        let width = (len as f64).sqrt() as usize;
        let height = width;
        if width * height != len {
            panic!("Invalid input string size");
        }
        Board::new(input, width, height)
    }

    pub fn gen_move(&self) -> String {
        String::from("Move result")
    }

    fn any_winner(&self) -> Option<u8> {
        for i in 0..self.height {
            for j in 0..self.width {
                for p in 1..3 {
                    for k in 0..4 {
                        let (count, _) = self.count_pos(p, i, j, k);
                        if count >= 5 {
                            return Some(p);
                        }
                    }
                }
            }
        }
        None
    }

    fn eval(&self, player: u8) -> u32 {
        let mut score: u32 = 0;
        for i in 0..self.height {
            for j in 0..self.width {
                for k in 0..4 {
                    score += 0;
                    let line = self.make_line(player, i, j, k);
                    score += line.score();
                }
            }
        }
        return score;
    }

    fn make_line(&self, player: u8, row: usize, col: usize, dir: usize) -> Line {
        let mut line = Line::new(0, 0, 0);
        let rev_dirs = vec![vec![0, -1], vec![-1, 0], vec![-1, -1], vec![1, -1]];
        let p_row = row as i32 + rev_dirs[dir][0];
        let p_col = col as i32 + rev_dirs[dir][1];
        let mut open_count = 2;
        if let Some(prev) = self.get(p_row, p_col) {
            if prev == player {
                // this position has been evaled
                return line;
            } else if prev != 0 {
                // prev position is placed by opponent player
                open_count -= 1;
            }
        } else {
            //prev position can not be placed
            open_count -= 1;
        }
        let (count, end_pos) = self.count_pos(player, row, col, dir);
        if let Some(p) = end_pos {
            if p != 0 && p != player {
                //end position can not be placed
                open_count -= 1;
            }
        } else {
            open_count -= 1;
        }
        line.count = count as usize;
        line.open_count = open_count;
        return line;
    }

    fn count_pos(&self, player: u8, row: usize, col: usize, dir: usize) -> (u32, Option<u8>) {
        let dirs = vec![vec![0, 1], vec![1, 0], vec![1, 1], vec![-1, 1]];
        let cur = &dirs[dir];
        let mut i = row as i32;
        let mut j = col as i32;
        let mut count = 0;
        let mut next_pos = None;
        loop {
            if let Some(p) = self.get(i, j) {
                if p != player {
                    next_pos = Some(p);
                    break;
                } else {
                    count += 1;
                    i += cur[0];
                    j += cur[1];
                }
            } else {
                break;
            }
        }
        (count, next_pos)
    }

    fn get(&self, row: i32, col: i32) -> Option<u8> {
        if !self.valid_pos(row, col) {
            None
        } else {
            Some(self.digits[row as usize][col as usize])
        }
    }

    fn valid_pos(&self, row: i32, col: i32) -> bool {
        row >= 0 && row < self.height as i32 && col >= 0 && col < self.width as i32
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_board_create() {
        let board = Board::new(String::from("121211"), 1, 6);
        assert_eq!(board.width, 1);
        assert_eq!(board.height, 6);
    }

    #[test]
    fn test_line() {
        let mut line = Line::new(5, 0, 0);
        assert_eq!(line.is_non_refutable(), true);

        line = Line::new(5, 0, 1);
        assert_eq!(line.is_non_refutable(), true);

        line = Line::new(5, 1, 0);
        assert_eq!(line.is_non_refutable(), false);
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
        assert_eq!(board.digits[0][0], 0);
        assert_eq!(board.digits[0][1], 0);
        assert_eq!(board.digits[1][4], 1);
        assert_eq!(board.digits[1][5], 2);
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

        board = Board::new(String::from("10000 10001 01021 00001 00001 00001"), 5, 6);
        assert_eq!(board.any_winner(), Some(1));
    }

    #[test]
    fn test_board_score() {
        let mut board = Board::new(String::from("1111000000"), 5, 2);
        assert_eq!(board.eval(2), 0);
        assert_eq!(board.eval(1), 1500);

        board = Board::new(String::from("1111111111"), 5, 2);
        assert_eq!(board.eval(2), 0);
        assert_eq!(board.eval(1), 20000);

        board = Board::new(String::from("10000 01000 00100"), 5, 3);
        assert_eq!(board.eval(1), 100);

        board = Board::new(String::from("10000 01000 00100 00000"), 5, 4);
        assert_eq!(board.eval(1), 500);
    }
}
