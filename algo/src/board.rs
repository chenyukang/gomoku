#[derive(Debug)]

pub struct Board {
    width: usize,
    height: usize,
    digits: Vec<Vec<u8>>
}

impl Board { 
    pub fn new(input: String, width: usize, height: usize) -> Self {
        let rows: Vec<u8> = input.chars().filter(|&e| e == '0' || e == '1' || e == '2')
                            .map(|e| e as u8 - '0' as u8).collect();
        if rows.len() != width * height {
            panic!("Invalid board size with {}*{} <> {}", width, height, rows.len());
        }
        let res = Self { 
            width: width,
            height: height,
            digits: rows.chunks(width).map(|x| x.to_vec() ).collect()
        };
        let current = res.check();
        println!("element: {}", res.digits[0][0]);
        println!("current: {:?}", current);
        res
    }

    fn check(&self) -> Option<u8> {
        for i in 0..self.height {
            for j in 0..self.width {
                println!("{} {} => {}", i, j, self.digits[i][j]);
                if let Some(role) = self.check_pos(i, j, 1) {
                    return Some(role);
                }
                if let Some(role) = self.check_pos(i, j, 2) {
                    return Some(role);
                }
            }
        }
        None
    }

    fn check_pos(&self, row: usize, col: usize, role: u8) -> Option<u8> {
        let mut row_count = 0;
        let mut col_count = 0;
        let mut angle_count = 0;
        for i in row..self.height {
            if self.digits[i][col] == role {
                col_count += 1;
                if col_count >= 5 {
                    return Some(role);
                }
            } else { 
                break;
            }
        }
        for j in col..self.width {
            if self.digits[row][j] == role {
                row_count += 1;
                if row_count >= 5 { 
                    return Some(role);
                }
            } else {
                break;
            }
        }

        let mut i = row as i32;
        let mut j = col as i32;
        loop {
            if self.valid_pos(i, j) && self.digits[i as usize][j as usize] == role {
                angle_count += 1;
                i += 1;
                j += 1;
                if angle_count >= 5 {
                    return Some(role);
                }
            } else {
                break;
            }
        }

        i = row as i32;
        j = col as i32;
        angle_count = 0;
        loop {
            if self.valid_pos(i, j) && self.digits[i as usize][j as usize] == role {
                angle_count += 1;
                i += 1;
                j -= 1;
                if angle_count >= 5 {
                    return Some(role);
                }
            } else {
                break;
            }
        }
        None
    }

    fn valid_pos(&self, row: i32, col: i32) -> bool { 
        return row >= 0 && row < self.height as i32 && col >= 0 && col < self.width as i32;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_board_create() {
        let board = Board::new(String::from("12"), 1, 2);
        assert_eq!(board.width, 1);
        assert_eq!(board.height, 2);
    }

    #[test]
    #[should_panic(expected = "Invalid board size with 2*2 <> 2")]
    fn test_board_size_validation() {
        Board::new(String::from("12"), 2, 2);
    }

    #[test]
    fn test_board_elements() {
        let board = Board::new(String::from("000112"), 3, 2);
        assert_eq!(board.digits[0][0], 0);
        assert_eq!(board.digits[0][1], 0);
        assert_eq!(board.digits[1][0], 1);
        assert_eq!(board.digits[1][2], 2);
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