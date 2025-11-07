pub mod cfg {
    pub static DIRS: &[&[i32; 2]; 4] = &[&[0, 1], &[1, 0], &[1, 1], &[-1, 1]];

    pub static REV_DIRS: &[&[i32; 2]; 4] = &[&[0, -1], &[-1, 0], &[-1, -1], &[1, -1]];

    pub static ALL_DIRS: &[&[i32; 2]; 8] = &[
        &[0, 1],
        &[1, 0],
        &[1, 1],
        &[-1, 1],
        &[0, -1],
        &[-1, 0],
        &[-1, -1],
        &[1, -1],
    ];

    pub fn opponent(player: u8) -> u8 {
        match player {
            1 => 2,
            2 => 1,
            _ => panic!("error: {}", player),
        }
    }
}

// Board configuration defaults. Change these to switch between Gomoku (15x15, connect5)
// and Connect4-style (6x7, connect4).
pub const BOARD_WIDTH: usize = 15;
pub const BOARD_HEIGHT: usize = 15;
#[cfg(test)]
pub const WIN_LEN: usize = 5;

// pub const BOARD_WIDTH: usize = 7;
// pub const BOARD_HEIGHT: usize = 6;
// #[cfg(test)]
// pub const WIN_LEN: usize = 4;
