#![allow(dead_code)]
use super::utils;
use std::cmp;

#[derive(Debug)]
pub struct Board {
    pub width: usize,
    pub height: usize,
    digits: Vec<Vec<u8>>,
}

#[derive(Debug)]
struct Line {
    count: u32,
    hole_count: i32,
    open_count: i32,
}

impl Line {
    pub fn new(count: u32, hole_count: i32, open_count: i32) -> Self {
        Self {
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
            (5, 0, _) => 100000,
            (4, 0, 2) => 9900,
            (4, 1, 2) => 2000,
            (3, 0, 2) => 2000,
            (5, 1, _) => 1000,
            (4, 0, 1) => 800,
            (4, 1, 0) => 100,
            (4, 1, 1) => 1000,
            (3, 0, 1) => 500,
            (3, 0, 0) => 100,
            (3, 1, 2) => 300,
            (3, 1, 0) => 80,
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

    pub fn any_winner(&self) -> Option<u8> {
        for i in 0..self.height {
            for j in 0..self.width {
                for p in 1..3 {
                    for k in 0..4 {
                        let (count, _, _) = self.count_pos(p, i, j, k, 0);
                        if count >= 5 {
                            return Some(p);
                        }
                    }
                }
            }
        }
        None
    }

    pub fn eval(&self, player: u8) -> i32 {
        let mut score: i32 = 0;
        for i in 0..self.height {
            for j in 0..self.width {
                for k in 0..4 {
                    score += 0;
                    let line1 = self.make_line(player, i, j, k, 0);
                    let line2 = self.make_line(player, i, j, k, 1);
                    let score1 = line1.score();
                    let score2 = line2.score();
                    score += cmp::max(score1 as i32, score2 as i32);
                }
            }
        }
        //println!("score: {}", score);
        return score;
    }

    // TBD: open direction should consider board width and height
    fn make_line(&self, player: u8, row: usize, col: usize, dir: usize, allow_hole: i32) -> Line {
        let mut open_count = 2;
        match self.get_prev(row as i32, col as i32, dir) {
            Some(v) => {
                if v == player {
                    return Line::new(0, 0, 0);
                } else if v != 0 {
                    open_count -= 1;
                }
            }
            _ => open_count -= 1,
        }
        let (count, hole_count, end_pos) = self.count_pos(player, row, col, dir, allow_hole);
        if end_pos.is_none() || end_pos == Some(utils::opponent(player)) {
            open_count -= 1;
        }
        Line::new(count, hole_count, open_count)
    }

    fn count_pos(
        &self,
        player: u8,
        row: usize,
        col: usize,
        dir: usize,
        allow_hole: i32,
    ) -> (u32, i32, Option<u8>) {
        let dirs = vec![vec![0, 1], vec![1, 0], vec![1, 1], vec![-1, 1]];
        let cur = &dirs[dir];
        let mut i = row as i32;
        let mut j = col as i32;
        let mut count = 0;
        let mut next_pos = None;
        let mut hole_count = allow_hole;
        loop {
            let nxt = self.get(i, j);
            if nxt == Some(player) || (count > 0 && hole_count > 0 && nxt == Some(0)) {
                count += 1;
                i += cur[0];
                j += cur[1];
                if nxt == Some(0) {
                    hole_count -= 1;
                }
            } else if nxt.is_none() {
                break;
            } else {
                next_pos = nxt;
                break;
            }
        }
        if allow_hole > 0 && hole_count == 0 && self.get_prev(i, j, dir) != Some(player) {
            count -= 1;
        }
        (count, allow_hole - hole_count, next_pos)
    }

    pub fn get(&self, row: i32, col: i32) -> Option<u8> {
        if !self.valid_pos(row, col) {
            None
        } else {
            Some(self.digits[row as usize][col as usize])
        }
    }

    fn get_prev(&self, row: i32, col: i32, dir: usize) -> Option<u8> {
        let rev_dirs = vec![vec![0, -1], vec![-1, 0], vec![-1, -1], vec![1, -1]];
        let p_row = row as i32 + rev_dirs[dir][0];
        let p_col = col as i32 + rev_dirs[dir][1];
        self.get(p_row, p_col)
    }

    fn valid_pos(&self, row: i32, col: i32) -> bool {
        row >= 0 && row < self.height as i32 && col >= 0 && col < self.width as i32
    }

    pub fn place(&mut self, row: usize, col: usize, player: u8) {
        self.digits[row][col] = player
    }

    pub fn is_remote_cell(&self, row: usize, col: usize) -> bool {
        let dirs = vec![
            vec![0, 1],
            vec![1, 0],
            vec![1, 1],
            vec![-1, 1],
            vec![0, -1],
            vec![-1, 0],
            vec![-1, -1],
            vec![1, -1],
        ];
        for d in 1..2 {
            for k in 0..8 {
                let p = self.get(row as i32 + dirs[k][0] * d, col as i32 + dirs[k][1] * d);
                if !p.is_none() && p != Some(0) {
                    return false;
                }
            }
        }
        true
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
        assert_eq!(board.eval(1), 800);

        board = Board::new(String::from("1111111111"), 5, 2);
        assert_eq!(board.eval(2), 0);
        assert_eq!(board.eval(1), 20000);

        board = Board::new(String::from("10000 01000 00100"), 5, 3);
        assert_eq!(board.eval(1), 100);

        board = Board::new(String::from("10000 01000 00100 00000"), 5, 4);
        assert_eq!(board.eval(1), 500);

        board = Board::new(String::from("10000 01100 00100 00000"), 5, 4);
        assert_eq!(board.eval(1), 660);

        board = Board::new(String::from("00000 01110 01110 01110 00000"), 5, 5);
        assert_eq!(board.eval(1), 16320);

        board = Board::new(String::from("101100 000000"), 6, 2);
        assert_eq!(board.eval(1), 1080);

        board = Board::new(String::from("101110 000000"), 6, 2);
        assert_eq!(board.eval(1), 3000);
    }
}
