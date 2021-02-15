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
