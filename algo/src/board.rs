#![allow(dead_code)]



#[derive(Debug)]
pub struct Board {
    width: usize,
    height: usize,
    digits: Vec<Vec<u8>>,
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

    fn check(&self) -> Option<u8> {
        for i in 0..self.height {
            for j in 0..self.width {
                for p in 1..3 { 
                    for k in 0..4 { 
                        let count = self.try_eval_pos(p, i, j, k);
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
                    score += self.try_eval_pos(player, i, j, k);
                }
            }
        }
        return score;
    }

    fn try_eval_pos(&self, player: u8, row: usize, col: usize, dir: usize) -> u32 { 
        let dirs = vec![vec![0, 1], vec![1, 0], vec![1, 1], vec![-1, 1]];
        let cur = &dirs[dir];
        let mut i = row as i32;
        let mut j = col as i32;
        let mut count = 0;
        loop { 
            if let Some(p) = self.get(i, j) { 
                if p != player {
                    break
                } else {
                    count += 1;
                    i += cur[0];
                    j += cur[1];
                }
            } else { 
                break;
            }
        }
        count
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
    fn test_board_check() {
        let mut board = Board::new(String::from("1111100000"), 5, 2);
        assert_eq!(board.check(), Some(1));

        board = Board::new(String::from("1111000000"), 5, 2);
        assert_eq!(board.check(), None);

        board = Board::new(String::from("1111022222"), 5, 2);
        assert_eq!(board.check(), Some(2));

        board = Board::new(String::from("111101111011110"), 5, 3);
        assert_eq!(board.check(), None);

        board = Board::new(String::from("1111011110111101111010000"), 5, 5);
        assert_eq!(board.check(), Some(1));

        board = Board::new(String::from("1111011110111100111010000"), 5, 5);
        assert_eq!(board.check(), None);

        board = Board::new(String::from("10000 01000 00100 00010 00001"), 5, 5);
        assert_eq!(board.check(), Some(1));

        board = Board::new(String::from("10000 01000 00000 00010 00001"), 5, 5);
        assert_eq!(board.check(), None);

        board = Board::new(String::from("22220 10000 01000 00100 00010 00001"), 5, 6);
        assert_eq!(board.check(), Some(1));

        board = Board::new(String::from("22220 00001 00010 00100 01000 10000"), 5, 6);
        assert_eq!(board.check(), Some(1));

        board = Board::new(String::from("22222 00001 00010 00100 01000 10000"), 5, 6);
        assert_eq!(board.check(), Some(2));

        board = Board::new(String::from("10000 10001 01021 00001 00001 00001"), 5, 6);
        assert_eq!(board.check(), Some(1));
    }
}
