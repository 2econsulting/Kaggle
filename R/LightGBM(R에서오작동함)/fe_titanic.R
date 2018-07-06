# title : fe_titanic
# author : jacob 

fe_titanic <- function(data){
  
  # nRooms
  data$nRooms <- sapply(strsplit(as.character(data$Cabin)," "), length)
  data$nRooms[data$nRooms==0] <- -9999
  
  # number
  data$number <- sapply(strsplit(as.character(data$Cabin)," "), function(x) gsub("[0-9]","",x[1]))
  data$number[is.na(data$number)] <- -9999
  data$number <- as.factor(data$number)
  
  # letter
  data$letter <- sapply(strsplit(as.character(data$Cabin)," "), function(x) gsub("[^0-9]","",x[1]))
  data$letter[is.na(data$letter)] <- -9999
  data$letter <- as.factor(data$letter)
  
  # FamilySize 
  data$FamilySize <- data$SibSp + data$Parch + 1 
  
  # FamilyRatio
  data$FamilyRatio <- (data$Parch + 1) / (data$SibSp + 1)
  
  # Adult
  data$Adult <- data$Age > 18
  data$Adult[is.na(data$Adult)] <- FALSE 
  data$Adult <- as.factor(data$Adult)
  
  # FamilySized
  # data$FamilySized[data$FamilySize == 1] <- 'Single' 
  # data$FamilySized[data$FamilySize < 5 & data$FamilySize >= 2] <- 'Small' 
  # data$FamilySized[data$FamilySize >= 5] <- 'Big' 
  # data$FamilySized <- as.factor(data$FamilySized)

  # isalone
  # data$isalone <- ifelse(data$FamilySize == 1, T, F)
  # data$isalone <- as.factor(data$isalone)
  
  # Surname & SurnameFreq
  data$Surname <- sapply(strsplit(gsub("[.,]","",data$Name), " "), function(x) x[1])
  data$Surname <- as.factor(data$Surname)
  # SurnameTable <- setNames(as.data.frame(table(data$Surname)),c("Surname","SurnameFreq"))
  # data <- merge(data, SurnameTable, by="Surname", all.x=TRUE)
  
  # title
  data$title <- gsub("^.*, (.*?)\\..*$", "\\1", data$Name)
  data$title[data$title == 'Capt'] <- 'Officer' 
  data$title[data$title == 'Col'] <- 'Officer' 
  data$title[data$title == 'Major'] <- 'Officer'
  data$title[data$title == 'Jonkheer'] <- 'Sir'
  data$title[data$title == 'Don'] <- 'Sir'  
  data$title[data$title == 'Sir'] <- 'Sir'
  data$title[data$title == 'Dr'] <- 'Dr'
  data$title[data$title == 'Rev'] <- 'Rev'
  data$title[data$title == 'the Countess'] <- 'Lady'
  data$title[data$title == 'Dona'] <- 'Lady'
  data$title[data$title == 'Mme'] <- 'Mrs' 
  data$title[data$title == 'Mlle'] <- 'Miss'
  data$title[data$title == 'Ms'] <- 'Mrs'
  data$title[data$title == 'Mr'] <- 'Mr'
  data$title[data$title == 'Mrs'] <- 'Mrs' 
  data$title[data$title == 'Miss'] <- 'Miss'
  data$title[data$title == 'Master'] <- 'Master'
  data$title[data$title == 'Lady'] <- 'Lady'
  data$title[!data$title %in% c("Officer","Sir","Dr","Rev","Lady","Mrs","Miss","Mr","Master")] <- "ohters"
  data$title <- as.factor(data$title)

  # nameLength
  # data$nameLength <- sapply(as.character(data$Name),function(x) nchar(x))
  # data$nameLength <- as.numeric(data$nameLength)

  # TicketSize
  # TicketTable <- setNames(as.data.frame(table(data$Ticket)),c("Ticket","TicketSize"))
  # data <- merge(data, TicketTable, by="Ticket", all.x=TRUE)
  # data$TicketSize <- as.numeric(data$TicketSize)
  
  # TicketSized
  # data$TicketSized[data$TicketSize == 1]   <- 'Single'
  # data$TicketSized[data$TicketSize < 5 & data$TicketSize>= 2]   <- 'Small'
  # data$TicketSized[data$TicketSize >= 5]   <- 'Big'
  # data$TicketSized <- as.factor(data$TicketSized)
  
  # AgeBin
  # data$AgeBin <- ifelse(data$Age<18,0, ifelse(data$Age<30,1, ifelse(data$Age<50,2,3)))
  # data$AgeBin <- as.factor(data$AgeBin)
  
  # Fare transformation 
  # data$FareDemean <- data$Fare - mean(data$Fare, na.rm=TRUE)
  # data$FareLog <- log(data$Fare)
  
  # hasCabin
  # data$hasCabin <- ifelse(data$Cabin!="",T,F)
  # data$hasCabin <- as.factor(data$hasCabin)
  
  # convert type
  data$Sex <- as.factor(data$Sex)
  data$Embarked <- as.factor(data$Embarked)
  data$Pclass <- as.factor(data$Pclass)
  data$SibSp <- as.numeric(data$SibSp)
  data$Parch <- as.numeric(data$Parch)
  
  # impute 
  data$Embarked[data$Embarked==""] <- 'S'
  data$Fare[is.na(data$Fare)] <- median(data$Fare, na.rm=TRUE)

  #data$Age[is.na(data$Age)] <- median(data$Age, na.rm=TRUE)
  #data$Age[is.na(data$Age)] <- -9999
  
  # remove 
  data$Age <- NULL
  data$Ticket <- NULL
  data$Cabin <- NULL
  data$PassengerId <- NULL
  data$Name <- NULL
  
  return(data)
}




