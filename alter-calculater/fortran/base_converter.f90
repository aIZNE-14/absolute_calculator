program base_converter
  use, intrinsic :: iso_fortran_env, only: int64, error_unit, input_unit, output_unit
  implicit none

  character(len=*), parameter :: digits = '0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz+/'
  character(len=256) :: input
  character(len=256) :: output
  integer :: source_base, target_base, io_status, index, digit, output_length
  integer(int64) :: value
  logical :: negative

  read(input_unit, *, iostat=io_status) source_base, target_base, input
  if (io_status /= 0) then
    write(error_unit, '(a)') 'Usage: enter source-base target-base integer'
    stop 2
  end if
  if (.not. supported(source_base) .or. .not. supported(target_base)) then
    write(error_unit, '(a)') 'Bases must be 2, 8, 10, 16, 32, or 60.'
    stop 2
  end if

  call parse_value(trim(input), source_base, value, negative, io_status)
  if (io_status /= 0) then
    write(error_unit, '(a)') 'Invalid digit or integer exceeds int64 range.'
    stop 1
  end if
  call format_value(value, target_base, output, output_length)
  if (negative .and. value /= 0_int64) then
    write(output_unit, '(a)', advance='no') '-'
  end if
  write(output_unit, '(a)') output(:output_length)

contains

  logical function supported(base)
    integer, intent(in) :: base
    supported = base >= 2 .and. base <= 64
  end function supported

  subroutine parse_value(text, base, result, is_negative, status)
    character(len=*), intent(in) :: text
    integer, intent(in) :: base
    integer(int64), intent(out) :: result
    logical, intent(out) :: is_negative
    integer, intent(out) :: status
    integer :: position, current_digit, text_length
    character :: character

    result = 0_int64
    is_negative = .false.
    status = 0
    text_length = len_trim(text)
    position = 1
    if (text_length == 0) then
      status = 1
      return
    end if
    if (text(1:1) == '-' .or. text(1:1) == '+') then
      if (text(1:1) == '-' .or. base < 63) then
        is_negative = text(1:1) == '-'
        position = 2
      end if
    end if
    if (position > text_length) then
      status = 1
      return
    end if
    do while (position <= text_length)
      character = text(position:position)
      current_digit = index(digits, character) - 1
      if (current_digit < 0 .and. base <= 36) then
        current_digit = index(digits, upper(character)) - 1
      end if
      if (current_digit < 0 .or. current_digit >= base) then
        status = 1
        return
      end if
      if (result > (huge(result) - current_digit) / base) then
        status = 1
        return
      end if
      result = result * base + current_digit
      position = position + 1
    end do
  end subroutine parse_value

  function upper(character) result(converted)
    character, intent(in) :: character
    character :: converted
    integer :: code
    converted = character
    code = iachar(character)
    if (code >= iachar('a') .and. code <= iachar('z')) converted = achar(code - 32)
  end function upper

  subroutine format_value(number, base, buffer, used)
    integer(int64), intent(in) :: number
    integer, intent(in) :: base
    character(len=*), intent(out) :: buffer
    integer, intent(out) :: used
    integer(int64) :: remaining
    integer :: remainder
    character(len=256) :: reversed

    remaining = number
    used = 0
    if (remaining == 0_int64) then
      buffer = '0'
      used = 1
      return
    end if
    do while (remaining > 0_int64)
      remainder = int(mod(remaining, int(base, int64)))
      used = used + 1
      reversed(used:used) = digits(remainder + 1:remainder + 1)
      remaining = remaining / base
    end do
    buffer = ' '
    do remainder = 1, used
      buffer(remainder:remainder) = reversed(used - remainder + 1:used - remainder + 1)
    end do
  end subroutine format_value

end program base_converter
